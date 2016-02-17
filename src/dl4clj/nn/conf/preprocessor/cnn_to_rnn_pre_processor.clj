(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/preprocessor/CnnToRnnPreProcessor.html"}
  dl4clj.nn.conf.preprocessor.cnn-to-rnn-pre-processor
  (:require [dl4clj.nn.conf.InputPreProcessor :refer :all])
  (:import [org.deeplearning4j.nn.conf.preprocessor CnnToRnnPreProcessor]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defn cnn-to-rnn-pre-processor [input-height, input-width,  num-Channels]
  (CnnToRnnPreProcessor. input-height input-width  num-Channels))

(defmethod pre-process CnnToRnnPreProcessor [^CnnToRnnPreProcessor this ^INDArray input batch-size]
  (.preProcess this input batch-size))

(defmethod backprop CnnToRnnPreProcessor [^CnnToRnnPreProcessor this ^INDArray output batch-size]
  (.backprop this output batch-size))
