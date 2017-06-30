(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/InputPreProcessor.html"}
    dl4clj.nn.conf.input-pre-processor
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many? array-of]])
  (:import [org.deeplearning4j.nn.conf InputPreProcessor]
           [org.deeplearning4j.nn.conf.inputs InputType]
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
  (BinomialSamplingPreProcessor.))

(defmethod pre-processors :unit-variance-processor [opts]
  (UnitVarianceProcessor.))

(defmethod pre-processors :rnn-to-cnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:rnn-to-cnn-pre-processor opts)]
    (RnnToCnnPreProcessor. input-height input-width num-channels)))

(defmethod pre-processors :zero-mean-and-unit-variance-pre-processor [opts]
  (ZeroMeanAndUnitVariancePreProcessor.))

(defmethod pre-processors :zero-mean-pre-pre-processor [opts]
  (ZeroMeanPrePreProcessor.))

(defmethod pre-processors :cnn-to-feed-forward-pre-processor [opts]
  (let [conf (:cnn-to-feed-forward-pre-processor opts)
        {input-height :input-height
         input-width :input-width
         num-channels :num-channels} conf]
    (cond (contains-many? conf :input-height :input-width :num-channels)
          (CnnToFeedForwardPreProcessor. input-height input-width num-channels)
          (contains-many? conf :input-height :input-width)
          (CnnToFeedForwardPreProcessor. input-height input-width)
          :else
          (CnnToFeedForwardPreProcessor.))))

(defmethod pre-processors :cnn-to-rnn-pre-processor [opts]
  (let [{input-height :input-height
         input-width :input-width
         num-channels :num-channels} (:cnn-to-rnn-pre-processor opts)]
    (CnnToRnnPreProcessor. input-height input-width num-channels)))

(defmethod pre-processors :feed-forward-to-cnn-pre-processor [opts]
  (let [conf (:feed-forward-to-cnn-pre-processor opts)
        {input-height :input-height
         input-width :input-width
         num-channels :num-channels} conf]
    (if (contains-many? conf :input-height :input-width :num-channels)
      (FeedForwardToCnnPreProcessor. input-height input-width num-channels)
      (FeedForwardToCnnPreProcessor. input-width input-height))))

(defmethod pre-processors :rnn-to-feed-forward-pre-processor [opts]
  (RnnToFeedForwardPreProcessor.))

(defmethod pre-processors :feed-forward-to-rnn-pre-processor [opts]
  (FeedForwardToRnnPreProcessor.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-binominal-sampling-pre-processor
  "Binomial sampling pre processor"
  []
  (pre-processors {:binominal-sampling-pre-processor {}}))

(defn new-unit-variance-processor
  "Unit variance operation"
  []
  (pre-processors {:unit-variance-processor {}}))

(defn new-rnn-to-cnn-pre-processor
  "A preprocessor to allow RNN and CNN layers to be used together

  args are:

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
  [& {:keys [input-height input-width num-channels]
      :as opts}]
  (pre-processors {:rnn-to-cnn-pre-processor opts}))

(defn new-zero-mean-and-unit-variance-pre-processor
  "Zero mean and unit variance operation"
  []
  (pre-processors {:zero-mean-and-unit-variance-pre-processor {}}))

(defn new-zero-mean-pre-pre-processor
  "Zero mean and unit variance operation"
  []
  (pre-processors {:zero-mean-pre-pre-processor {}}))

(defn new-cnn-to-feed-forward-pre-processor
  "A preprocessor to allow CNN and standard feed-forward network layers to be used together.
  For example, CNN -> Denselayer

  args are:

  :input-height (int), the input height

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
  [& {:keys [input-height input-width num-channels]
      :as opts}]
  (pre-processors {:cnn-to-feed-forward-pre-processor opts}))

(defn new-cnn-to-rnn-pre-processor
  "A preprocessor to allow CNN and RNN layers to be used together.

  args are:

  :input-height (int), the input height

  :input-width (int), the width of the input

  :num-channels (int), the number of channels

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
  [& {:keys [input-height input-width num-channels]
      :as opts}]
  (pre-processors {:cnn-to-rnn-pre-processor opts}))

(defn new-feed-forward-to-cnn-pre-processor
  "A preprocessor to allow CNN and standard feed-forward network layers to be used together.

  args are:

  :input-height (int), the input height

  :input-width (int), the width of the input

  :num-channels (int), the number of channels

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
  [& {:keys [input-height input-width num-channels]
      :as opts}]
  (pre-processors {:feed-forward-to-cnn-pre-processor opts}))

(defn new-rnn-to-feed-forward-pre-processor
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

  []
  (pre-processors {:rnn-to-feed-forward-pre-processor {}}))

(defn new-feed-forward-to-rnn-pre-processor
  "A preprocessor to allow RNN and feed-forward network layers to be used together.

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
  []
  (pre-processors {:feed-forward-to-rnn-pre-processor {}}))

(defn new-composable-input-pre-processor
  "allows you to combine multiple pre-processors into a single pre-processor

  :pre-processors (coll), a collection of pre-processor objects
   - can be pre-processors built using the above fns or a configuration map
     that is passed to the pre-processors multi method

  ie. (composable-input-pre-processor
       :pre-processors [(zero-mean-pre-pre-processor)
                        (binominal-sampling-pre-processor)
                        {:cnn-to-feed-forward-pre-processor
                         {:input-height 2 :input-width 3 :num-channels 4}}])"
  [& {:keys [pre-processors]
      :as opts}]
  (let [pp (into []
                 (for [each pre-processors]
                   (if (map? each)
                     (pre-processors each)
                     each)))]
   (ComposableInputPreProcessor.
   (array-of :data pp :java-type InputPreProcessor))))
