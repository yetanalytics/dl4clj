(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.html"}
  dl4clj.nn.api.layer
  (:require [dl4clj.nn.api.model :as model])
  (:import [org.deeplearning4j.nn.api Layer]))

(defmethod model/num-params Layer
  ([^Layer l] 
   (.numParams l))
  ([^Layer l x] 
   (.numParams l x)))




