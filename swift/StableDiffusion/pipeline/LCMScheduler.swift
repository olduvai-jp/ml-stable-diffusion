import Accelerate
import CoreML

struct Config {
    let timestepScaling: Float
    let predictionType: String
    let thresholding: Bool
    let clipSample: Bool
    let clipSampleRange: Float
    let originalInferenceSteps: Int
    let trainTimeStepCount: Int
    let seed: UInt32
}

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers LCMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py)
///
@available(iOS 16.2, macOS 13.1, *)
public final class LCMScheduler: Scheduler {
    private let order = 1
    private let config: Config
    private var stepIndex: Int?
    public let trainStepCount: Int
    public var inferenceStepCount: Int
    private var customTimeSteps = false
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    private let finalAlphaCumProd: Float
    public let initNoiseSigma: Double
    public var timeSteps: [Int]

    public let alpha_t: [Float] = []
    public let sigma_t: [Float] = []
    public let lambda_t: [Float] = []
    
    public let solverOrder = 2
    private(set) var lowerOrderStepped = 0
    
    private var usingKarrasSigmas = false

    /// Whether to use lower-order solvers in the final steps. Only valid for less than 15 inference steps.
    /// We empirically find this trick can stabilize the sampling of DPM-Solver, especially with 10 or fewer steps.
    public let useLowerOrderFinal = true
    
    private var sigmaData: Float = 0.5

    // Stores solverOrder (2) items
    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    /// Create a scheduler that uses a second order DPM-Solver++ algorithm.
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    ///   - timeStepSpacing: How to space time steps
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        clipSample: Bool = false,
        clipSampleRange: Float = 1.0,
        setAlphaToOne: Bool = true,
        predictionType: String = "epsilon",
        thresholding: Bool = false,
        timeStepSpacing: TimeStepSpacing = .leading,
        timestepScaling: Float = 10.0
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        
        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map({ $0 * $0 })
        }
        
        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd
        finalAlphaCumProd = setAlphaToOne ? 1.0 : alphasCumProd[0]

        initNoiseSigma = 1.0

        timeSteps = (0..<trainStepCount).reversed()

        config = Config(
            timestepScaling: timestepScaling,
            predictionType: predictionType,
            thresholding: thresholding,
            clipSample: clipSample,
            clipSampleRange: clipSampleRange,
            originalInferenceSteps: stepCount,
            trainTimeStepCount: trainStepCount,
            seed: 12345
        )
        setTimeSteps(inferenceStepCount: stepCount)
    }

    private func initStepIndex(_ timeStep: Int) {
        let indexCandidates = timeSteps.enumerated().compactMap {
            $0.1 == timeStep ? $0.0 : nil
        }
        if indexCandidates.count > 1 {
            stepIndex = indexCandidates[1]
        } else {
            stepIndex = indexCandidates[0]
        }
    }

    func setTimeSteps(
        inferenceStepCount: Int? = nil,
        originalInferenceSteps: Int? = nil,
        timeSteps: [Int]? = nil,
        strength: Float = 1.0
    ) {
        let originalSteps = originalInferenceSteps ?? config.originalInferenceSteps
        assert(originalSteps <= config.trainTimeStepCount)

        let k = config.trainTimeStepCount / originalSteps
        let lcmOriginTimeSteps = (1..<Int(Float(originalSteps) * strength) + 1).map { $0 * k - 1 }

        if let _timeSteps = timeSteps {
            let trainTimeSteps = Set(lcmOriginTimeSteps)
            var nonTrainTimeSteps = [Int]()
            for i in 1..<_timeSteps.endIndex {
                if _timeSteps[i] >= _timeSteps[i - 1] {
                    fatalError("`custom_timesteps` must be in descending order.")
                }
                if !trainTimeSteps.contains(_timeSteps[i]) {
                    nonTrainTimeSteps.append(_timeSteps[i])
                }
            }

            let inferenceStepCount = _timeSteps.count
            customTimeSteps = true

            let initTimeStep = min(Int(Float(inferenceStepCount) * strength), inferenceStepCount)
            let tStart = max(inferenceStepCount - initTimeStep, 0)
            self.timeSteps = Array(_timeSteps[(tStart * order)...])
        } else {
            if inferenceStepCount! > config.trainTimeStepCount {
                fatalError(
                    "`num_inference_steps`: \(inferenceStepCount!) cannot be larger than `self.config.train_timesteps`:" +
                    " \(config.trainTimeStepCount) as the unet model trained with this scheduler can only handle" +
                    " maximal \(config.trainTimeStepCount) timesteps."
                )
            }

            let skippingStep = lcmOriginTimeSteps.count // num_inference_steps

            self.inferenceStepCount = inferenceStepCount!

            let lcmOriginTimeSteps = lcmOriginTimeSteps.reversed()
            let inferenceIndices = stride(from: 0.0, to: Double(lcmOriginTimeSteps.count), by: Double(lcmOriginTimeSteps.count) / Double(inferenceStepCount! + 1)).map { Int($0) }
            self.timeSteps = lcmOriginTimeSteps.enumerated().compactMap { inferenceIndices.contains($0) ? $1 : nil }
        }
        stepIndex = nil
    }
    
    private func getScalingsForBoundaryConditionDiscrete(_ timestep: Int) -> (Float, Float) {
        sigmaData = 0.5  // Default: 0.5
        let scaledTimestep = Float(timestep) * config.timestepScaling

        let cSkip = pow(sigmaData, 2) / (pow(scaledTimestep, 2) + pow(sigmaData,2))
        let cOut = scaledTimestep / pow(pow(scaledTimestep, 2) + pow(sigmaData, 2), 0.5)
        return (cSkip, cOut)
    }

    public func step(output: MLShapedArray<Float32>, timeStep t: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        if stepIndex == nil {
            initStepIndex(t)
        }
        let prevTimestep = stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex! + 1]

        let alphaProdT = alphasCumProd[t]
        let alphaProdTPrev = prevTimestep >= 0 ? alphasCumProd[prevTimestep] : finalAlphaCumProd

        let betaProdT = 1 - alphaProdT
        let betaProdTPrev = 1 - alphaProdTPrev

        let (cSkip, cOut) = self.getScalingsForBoundaryConditionDiscrete(t)

        var predictedOriginalSample = output
        predictedOriginalSample.withUnsafeMutableShapedBufferPointer { (pt, shape, stride) in
            if config.predictionType == "epsilon" {
                sample.withUnsafeShapedBufferPointer { (s, _, _) in
                    for i in pt.indices {
                        pt[i] = (s[i] - sqrt(betaProdT) * pt[i]) / sqrt(alphaProdT)
                    }
                }
            } else if config.predictionType == "sample" {
                return
            } else if config.predictionType == "v_prediction" {
                sample.withUnsafeShapedBufferPointer { (s, _, _) in
                    for i in pt.indices {
                        pt[i] = sqrt(alphaProdT) * s[i] - sqrt(betaProdT) * pt[i]
                    }
                }
            } else {
                fatalError("prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction` for `LCMScheduler`.")
            }
        }

        if config.clipSample {
            predictedOriginalSample.withUnsafeMutableShapedBufferPointer { (pt, _, _) in
                for i in pt.indices {
                    pt[i] = min(max(pt[i], -config.clipSampleRange), config.clipSampleRange)
                }
            }
        }

        var denoised = predictedOriginalSample
        denoised.withUnsafeMutableShapedBufferPointer { (pt, _, _) in
            sample.withUnsafeShapedBufferPointer { (s, _, _) in
                for i in pt.indices {
                    pt[i] = cOut * pt[i] + cSkip * s[i]
                }
            }
        }

        var prevSample = denoised
        if stepIndex != inferenceStepCount - 1 {
            var random = TorchRandomSource(seed: config.seed)
            let noise = random.normalShapedArray(output.shape, mean: 0.0, stdev: 1.0)
            prevSample.withUnsafeMutableShapedBufferPointer { (pt, _, _) in
                noise.withUnsafeShapedBufferPointer { (n, _, _) in
                    for i in pt.indices {
                        pt[i] = sqrt(alphaProdTPrev) * pt[i] + sqrt(betaProdTPrev) * Float(n[i])
                    }
                }
            }
        } else {
            prevSample = denoised
        }

        stepIndex = stepIndex! + 1

        return prevSample
    }
}
