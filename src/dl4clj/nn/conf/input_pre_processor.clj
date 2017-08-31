(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/InputPreProcessor.html"}
    dl4clj.nn.conf.input-pre-processor
  (:require [dl4clj.utils :refer [generic-dispatching-fn array-of obj-or-code?]]
            [clojure.core.match :refer [match]])
  (:import [org.deeplearning4j.nn.conf InputPreProcessor]
           [org.deeplearning4j.nn.conf.inputs InputType]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.preprocessor BinomialSamplingPreProcessor
            ComposableInputPreProcessor UnitVarianceProcessor RnnToCnnPreProcessor
            ZeroMeanAndUnitVariancePreProcessor ZeroMeanPrePreProcessor
            CnnToFeedForwardPreProcessor CnnToRnnPreProcessor FeedForwardToCnnPreProcessor
            FeedForwardToRnnPreProcessor RnnToFeedForwardPreProcessor]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method hevy lifting (constructor calling)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti pre-processors
  "constructs nearly any pre-processor"
  generic-dispatching-fn)

(defmethod pre-processors :binominal-sampling-pre-processor [opts]
  `(BinomialSamplingPreProcessor.))

(defmethod pre-processors :unit-variance-processor [opts]
  `(UnitVarianceProcessor.))

(defmethod pre-processors :rnn-to-cnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:rnn-to-cnn-pre-processor opts)]
    `(RnnToCnnPreProcessor. ~input-height ~input-width ~num-channels)))

(defmethod pre-processors :zero-mean-and-unit-variance-pre-processor [opts]
  `(ZeroMeanAndUnitVariancePreProcessor.))

(defmethod pre-processors :zero-mean-pre-pre-processor [opts]
  `(ZeroMeanPrePreProcessor.))

(defmethod pre-processors :cnn-to-feed-forward-pre-processor [opts]
  (let [conf (:cnn-to-feed-forward-pre-processor opts)
        {input-height :input-height
         input-width :input-width
         num-channels :num-channels} conf]
    (match [conf]
           [{:input-height _ :input-width _ :num-channels _}]
           `(CnnToFeedForwardPreProcessor. ~input-height ~input-width ~num-channels)
           [{:input-height _ :input-width _}]
           `(CnnToFeedForwardPreProcessor. ~input-height ~input-width)
           :else
           `(CnnToFeedForwardPreProcessor.))))

(defmethod pre-processors :cnn-to-rnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:cnn-to-rnn-pre-processor opts)]
    `(CnnToRnnPreProcessor. ~input-height ~input-width ~num-channels)))

(defmethod pre-processors :feed-forward-to-cnn-pre-processor [opts]
  (let [conf (:feed-forward-to-cnn-pre-processor opts)
        {input-height :input-height
         input-width :input-width
         num-channels :num-channels} conf]
    (match [conf]
           [{:input-height _ :input-width _ :num-channels _}]
           `(FeedForwardToCnnPreProcessor. ~input-height ~input-width ~num-channels)
           :else
           `(FeedForwardToCnnPreProcessor. ~input-width ~input-height))))

(defmethod pre-processors :rnn-to-feed-forward-pre-processor [opts]
  `(RnnToFeedForwardPreProcessor.))

(defmethod pre-processors :feed-forward-to-rnn-pre-processor [opts]
  `(FeedForwardToRnnPreProcessor.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-binominal-sampling-pre-processor
  "Binomial sampling pre processor

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:binominal-sampling-pre-processor {}})]
    (obj-or-code? as-code? code)))

(defn new-unit-variance-processor
  "Unit variance operation

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:unit-variance-processor {}})]
    (obj-or-code? as-code? code)))

(defn new-rnn-to-cnn-pre-processor
  "A preprocessor to allow RNN and CNN layers to be used together

  args are:

  :as-code? (boolean), return java object or code for creating it

  :input-height (int), the input height

  :input-width (int), the width of the input

  :num-channels (int), the number of channels

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
  [& {:keys [input-height input-width num-channels as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:rnn-to-cnn-pre-processor opts})]
    (obj-or-code? as-code? code)))

(defn new-zero-mean-and-unit-variance-pre-processor
  "Zero mean and unit variance operation

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:zero-mean-and-unit-variance-pre-processor {}})]
    (obj-or-code? as-code? code)))

(defn new-zero-mean-pre-pre-processor
  "Zero mean and unit variance operation

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:zero-mean-pre-pre-processor {}})]
    (obj-or-code? as-code? code)))

(defn new-cnn-to-feed-forward-pre-processor
  "A preprocessor to allow CNN and standard feed-forward network layers to be used together.
  For example, CNN -> Denselayer

  args are:

  :input-height (int), the input height

  :as-code? (boolean), return java object or code for creating it

  :input-width (int), the width of the input

  :num-channels (int), the number of channels

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
  [& {:keys [input-height input-width num-channels as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:cnn-to-feed-forward-pre-processor opts})]
    (obj-or-code? as-code? code)))

(defn new-cnn-to-rnn-pre-processor
  "A preprocessor to allow CNN and RNN layers to be used together.

  args are:

  :input-height (int), the input height

  :input-width (int), the width of the input

  :num-channels (int), the number of channels

  :as-code? (boolean), return java object or code for creating it

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
  [& {:keys [input-height input-width num-channels as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:cnn-to-rnn-pre-processor opts})]
    (obj-or-code? as-code? code)))

(defn new-feed-forward-to-cnn-pre-processor
  "A preprocessor to allow CNN and standard feed-forward network layers to be used together.

  args are:

  :input-height (int), the input height

  :input-width (int), the width of the input

  :num-channels (int), the number of channels

  :as-code? (boolean), return java object or code for creating it

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
  [& {:keys [input-height input-width num-channels as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:feed-forward-to-cnn-pre-processor ~opts})]
    (obj-or-code? as-code? code)))

(defn new-rnn-to-feed-forward-pre-processor
  "A preprocessor to allow RNN and feed-forward network layers to be used together.

  For example, GravesLSTM -> OutputLayer or GravesLSTM -> DenseLayer

  :as-code? (boolean), return java object or code for creating it

  This does two things:
  (a) Reshapes activations out of RNN layer
  (which is 3D with shape [miniBatchSize,layerSize,timeSeriesLength])
  into 2d activations (with shape [miniBatchSize*timeSeriesLength,layerSize])
  suitable for use in feed-forward layers.

  (b) Reshapes 2d epsilons (weights*deltas from feed forward layer, with shape
  [miniBatchSize*timeSeriesLength,layerSize]) into 3d epsilons
  (with shape [miniBatchSize,layerSize,timeSeriesLength]) for use in RNN layer"

  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:rnn-to-feed-forward-pre-processor {}})]
    (obj-or-code? as-code? code)))

(defn new-feed-forward-to-rnn-pre-processor
  "A preprocessor to allow RNN and feed-forward network layers to be used together.

  :as-code? (boolean), return java object or code for creating it

  For example, DenseLayer -> GravesLSTM

  This does two things:
  (a) Reshapes activations out of FeedFoward layer
  (which is 2D with shape [miniBatchSize*timeSeriesLength,layerSize])
  into 3d activations (with shape [miniBatchSize,layerSize,timeSeriesLength])
  suitable to feed into RNN layers.

  (b) Reshapes 3d epsilons
  (weights*deltas from RNN layer, with shape [miniBatchSize,layerSize,timeSeriesLength])
  into 2d epsilons (with shape [miniBatchSize*timeSeriesLength,layerSize])
  for use in feed forward layer"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:feed-forward-to-rnn-pre-processor {}})]
    (obj-or-code? as-code? code)))

(defn new-composable-input-pre-processor
  "allows you to combine multiple pre-processors into a single pre-processor

  :components (coll), a collection of pre-processors
   - can be a mix of the above fns or a configuration map
     that is passed to the pre-processors multi method

  :as-code? (boolean), return java object or code for creating it

  ie. (composable-input-pre-processor
       :components [(zero-mean-pre-pre-processor)
                        (binominal-sampling-pre-processor)
                        {:cnn-to-feed-forward-pre-processor
                         {:input-height 2 :input-width 3 :num-channels 4}}])"
  [& {:keys [components as-code?]
      :or {as-code? true}
      :as opts}]
  (let [pp (into []
                 (for [each components]
                   (if (map? each)
                     (pre-processors each)
                     each)))
        pp-array `(array-of :data ~pp :java-type InputPreProcessor)
        code `(ComposableInputPreProcessor. ~pp-array)]
    (obj-or-code? as-code? code)))
