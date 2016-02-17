(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.hmtl"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier]))

(defmulti fit (fn [x & more] (type x)))
