(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/InputPreProcessor.html"}
    dl4clj.nn.conf.input-pre-processor
  (:require [dl4clj.nn.conf.constants :as constants])
  (:import [org.deeplearning4j.nn.conf InputPreProcessor]
           [org.deeplearning4j.nn.conf.inputs InputType]
           [org.deeplearning4j.nn.conf.preprocessor BinomialSamplingPreProcessor
            ComposableInputPreProcessor UnitVarianceProcessor RnnToCnnPreProcessor
            ZeroMeanAndUnitVariancePreProcessor ZeroMeanPrePreProcessor
            CnnToFeedForwardPreProcessor CnnToRnnPreProcessor FeedForwardToCnnPreProcessor
            FeedForwardToRnnPreProcessor RnnToFeedForwardPreProcessor]))

(defn backprop
  "Reverse the preProcess during backprop."
  [this output mini-batch-size]
  (.backprop this output (int mini-batch-size)))

(defn clone
  [this]
  (.clone this))

(defn pre-process
  "Pre preProcess input/activations for a multi layer network"
  [this input mini-batch-size]
  (.preProcess this input (int mini-batch-size)))

(defn feed-forward-mask-array
  [this mask-array current-mask-state mini-batch-size]
  (.feedForwardMaskArray this mask-array current-mask-state (int mini-batch-size)))

(defn get-output-type
  "For a given type of input to this preprocessor, what is the type of the output?

  The InputType class is used to track and define the types of activations etc used in a ComputationGraph.
  This is most useful for automatically adding preprocessors between layers, and automatically setting nIn values."
  [this input-type]
  (.getOutputType this input-type))

(defn pre-process-type
  [opts]
  (first (keys opts)))

(defn input-types
  [b opts]
  (let [{typez :type
         height :height
         width :width
         depth :depth
         size :size} opts]
    (cond
      (= typez :convolutional)
      (get-output-type b (InputType/convolutional height width depth))
      (= typez :convolutional-flat)
      (get-output-type b (InputType/convolutionalFlat height width depth))
      (= typez :feed-forward)
      (get-output-type b (InputType/feedForward size))
      (= typez :recurrent)
      (get-output-type b (InputType/recurrent size))
      :else b)))

(defn fn-calls
  "determines what functions to call based on opts map
  and makes the method call with appropriate params"
  [builder opts]
  (let [b builder]
    (if (contains? opts :backprop)
      (let [{output :output
             mini-batch-size :mini-batch-size} (:backprop opts)]
        (backprop b output mini-batch-size))
      b)
    (if (contains? opts :pre-process)
      (let [{input :input
             mini-batch-size :mini-batch-size} (:pre-process opts)]
        (pre-process b input mini-batch-size))
      b)
    (if (contains? opts :get-output-type)
      (input-types b opts)
      b)
    (if (contains? opts :feed-forward-mask-array)
      (let [{mask-array :mask-array
             current-mask-state :current-mask-state
             mini-batch-size :mini-batch-size} (:feed-forward-mask-array opts)]
        (feed-forward-mask-array b mask-array
                                 (constants/value-of {:mask-state current-mask-state})
                                 mini-batch-size))
      b)
    b))

(defmulti pre-processors
  "constructs nearly any pre-processor and can run its methods

  opts should look like:

  {:type-of-preprocessor
  {:input-height int (these first 3 keys are only needed when dealing with cnn preprocessors)
  :input-width int
  :num-channels int
  :backprop {:output INDarray :mini-batch-size int} (optional used for call to backprop)
  :pre-process {:input INDarray :mini-batch-size int} (optional used for call to pre-process)
  :get-output-type {:input-type (one of:)
                        {:type :convolutional
                         :height int :width int :depth int}
                        {:type :convolutional-flat
                         :height int :width int :depth in}
                        {:type :feed-forward :size int}
                        {:type :recurrent :size int}}}
  :feed-forward-mask-array {:mask-array INDArray
                            :current-mask-state keyword (either :active or :passthrough)
                            :mini-batch-size int}"
  pre-process-type)

(defmethod pre-processors :binominal-sampling-pre-processor [opts]
  (fn-calls (BinomialSamplingPreProcessor.)
            (:binominal-sampling-pre-processor opts)))

(defmethod pre-processors :unit-variance-processor [opts]
  ;; see if you cant find out how to set columnStds, default = null
  (fn-calls (UnitVarianceProcessor.) (:unit-variance-processor opts)))

(defmethod pre-processors :rnn-to-cnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:rnn-to-cnn-pre-processor opts)]
    (fn-calls (RnnToCnnPreProcessor. input-height input-width num-channels)
              (:rnn-to-cnn-pre-processor opts))))

(defmethod pre-processors :zero-mean-and-unit-variance-pre-processor [opts]
  (fn-calls (ZeroMeanAndUnitVariancePreProcessor.)
            (:zero-mean-and-unit-variance-pre-processor opts)))

(defmethod pre-processors :zero-mean-pre-pre-processor [opts]
  (fn-calls (ZeroMeanPrePreProcessor.) (:zero-mean-pre-pre-processor opts)))

(defmethod pre-processors :cnn-to-feed-forward-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:cnn-to-feed-forward-pre-processor opts)]
    (fn-calls
     (cond (every? nil? [input-height input-width num-channels])
          (CnnToFeedForwardPreProcessor.)
          (and (int? input-height) (int? input-width) (nil? num-channels))
          (CnnToFeedForwardPreProcessor. input-height input-width)
          (every? int? [input-height input-width num-channels])
          (CnnToFeedForwardPreProcessor. input-height input-width num-channels))
     (:cnn-to-feed-forward-pre-processor opts))))

(defmethod pre-processors :cnn-to-rnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:cnn-to-rnn-pre-processor opts)]
    (fn-calls (CnnToRnnPreProcessor. input-height input-width num-channels)
              (:cnn-to-rnn-pre-processor opts))))

(defmethod pre-processors :feed-forward-to-cnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:feed-forward-to-cnn-pre-processor opts)]
    (fn-calls
     (if (nil? num-channels)
       (FeedForwardToCnnPreProcessor. input-width input-height)
       (FeedForwardToCnnPreProcessor. input-height input-width num-channels))
     (:feed-forward-to-cnn-pre-processor opts))))

(defmethod pre-processors :rnn-to-feed-forward-pre-processor [opts]
  (fn-calls (RnnToFeedForwardPreProcessor.) (:rnn-to-feed-forward-pre-processor opts)))





(defn binominal-sampling-pre-processor
  "Binomial sampling pre processor"
  [opts]
  (pre-processors {:binominal-sampling-pre-processor opts}))

(defn unit-variance-processor
  "Unit variance operation"
  [opts]
  (pre-processors {:unit-variance-processor opts}))

(defn rnn-to-cnn-pre-processor
  "A preprocessor to allow RNN and CNN layers to be used together

  opts must include {:input-height int :input-width int :num-channels int}

  For example, time series (video) input -> ConvolutionLayer, or conceivable GravesLSTM -> ConvolutionLayer
   Functionally equivalent to combining RnnToFeedForwardPreProcessor + FeedForwardToCnnPreProcessor

  Specifically, this does two things:
  (a) Reshape 3d activations out of RNN layer, with shape
     [miniBatchSize, numChannels*inputHeight*inputWidth, timeSeriesLength]
     into 4d (CNN) activations (with shape [numExamples*timeSeriesLength, numChannels, inputWidth, inputHeight])

  (b) Reshapes 4d epsilons (weights.*deltas) out of CNN layer
      (with shape [numExamples*timeSeriesLength, numChannels, inputHeight, inputWidth])
      into 3d epsilons with shape [miniBatchSize, numChannels*inputHeight*inputWidth, timeSeriesLength]
      suitable to feed into CNN layers.
  Note: numChannels is equivalent to depth or featureMaps referenced in different literature"
  [opts]
  (pre-processors {:rnn-to-cnn-pre-processor opts}))

(defn zero-mean-and-unit-variance-pre-processor
  "Zero mean and unit variance operation"
  [opts]
  (pre-processors {:zero-mean-and-unit-variance-pre-processor opts}))

(defn zero-mean-pre-pre-processor
  "Zero mean and unit variance operation"
  [opts]
  (pre-processors {:zero-mean-pre-pre-processor opts}))

(defn cnn-to-feed-forward-pre-processor
  "A preprocessor to allow CNN and standard feed-forward network layers to be used together.
  For example, CNN -> Denselayer

  opts should include {:input-height int :input-width int :num-channels int} but are not required.
  the constructor accepts no args, :input-height and :input-width or all 3 params

  This does two things:

  (a) Reshapes epsilons (weights*deltas) out of FeedFoward layer
     (which is 2D or 3D with shape [numExamples, inputHeight*inputWidth*numChannels])
     into 4d epsilons (with shape [numExamples, numChannels, inputHeight, inputWidth])
     suitable to feed into CNN layers.
  (b) Reshapes 4d activations out of CNN layer,
      with shape [numExamples, numChannels, inputHeight, inputWidth]) into 2d activations
      (with shape [numExamples, inputHeight*inputWidth*numChannels])
      for use in feed forward layer

  Note: numChannels is equivalent to depth or featureMaps referenced in different literature"
  [opts]
  (pre-processors {:cnn-to-feed-forward-pre-processor opts}))

(defn cnn-to-rnn-pre-processor
  "A preprocessor to allow CNN and RNN layers to be used together.

  opts must include {:input-height int :input-width int :num-channels int}

  For example, ConvolutionLayer -> GravesLSTM Functionally equivalent to combining
  CnnToFeedForwardPreProcessor + FeedForwardToRnnPreProcessor

  Specifically, this does two things:
  (a) Reshape 4d activations out of CNN layer,
  with shape [timeSeriesLength*miniBatchSize, numChannels, inputHeight, inputWidth])
  into 3d (time series) activations (with shape [numExamples, inputHeight*inputWidth*numChannels, timeSeriesLength])
  for use in RNN layers

  (b) Reshapes 3d epsilons (weights.*deltas) out of RNN layer
  (with shape [miniBatchSize,inputHeight*inputWidth*numChannels,timeSeriesLength])
  into 4d epsilons with shape [miniBatchSize*timeSeriesLength, numChannels, inputHeight, inputWidth]
  suitable to feed into CNN layers.

  Note: numChannels is equivalent to depth or featureMaps referenced in different literature"
  [opts]
  (pre-processors {:cnn-to-rnn-pre-processor opts}))

(defn feed-forward-to-cnn-pre-processor
  "A preprocessor to allow CNN and standard feed-forward network layers to be used together.

  opts should include {:input-height int :input-width int :num-channels int} but are not required.
  the constructor accepts no args, :input-height and :input-width or all 3 params


  For example, DenseLayer -> CNN
  This does two things:
  (a) Reshapes activations out of FeedFoward layer (which is 2D or 3D with shape
  [numExamples, inputHeight*inputWidth*numChannels]) into 4d activations
  (with shape [numExamples, numChannels, inputHeight, inputWidth])
  suitable to feed into CNN layers.

  (b) Reshapes 4d epsilons (weights*deltas) from CNN layer, with shape
  [numExamples, numChannels, inputHeight, inputWidth]) into 2d epsilons
  (with shape [numExamples, inputHeight*inputWidth*numChannels])
  for use in feed forward layer

  Note: numChannels is equivalent to depth or featureMaps referenced in different literature"
  [opts]
  (pre-processors {:feed-forward-to-cnn-pre-processor opts}))

(defn rnn-to-feed-forward-pre-processor
  "A preprocessor to allow RNN and feed-forward network layers to be used together.

  For example, GravesLSTM -> OutputLayer or GravesLSTM -> DenseLayer

  This does two things:
  (a) Reshapes activations out of RNN layer
  (which is 3D with shape [miniBatchSize,layerSize,timeSeriesLength])
  into 2d activations (with shape [miniBatchSize*timeSeriesLength,layerSize])
  suitable for use in feed-forward layers.

  (b) Reshapes 2d epsilons (weights*deltas from feed forward layer, with shape
  [miniBatchSize*timeSeriesLength,layerSize]) into 3d epsilons
  (with shape [miniBatchSize,layerSize,timeSeriesLength]) for use in RNN layer"

  [opts]
  (pre-processors {:rnn-to-feed-forward-pre-processor opts}))
