(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/InputPreProcessor.html"}
  dl4clj.nn.conf.InputPreProcessor
  (:import [org.deeplearning4j.nn.conf InputPreProcessor]
           [org.nd4j.linalg.api.ndarray INDArray]))


(defmulti pre-process (fn [preprocessor ^INDArray input mini-batch-size] (type preprocessor)))
  
(defmulti backprop (fn [preprocessor ^INDArray output mini-batch-size] (type preprocessor)))
