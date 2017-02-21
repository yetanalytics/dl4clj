(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/optimize/stepfunctions/DefaultStepFunction.html"}
  dl4clj.optimize.step-functions.default-step-function
  (:import [org.deeplearning4j.optimize.stepfunctions DefaultStepFunction]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defn default-step-function []
  (DefaultStepFunction.))

(defn step
  ([^DefaultStepFunction sf]
   (.step sf))
  ([^DefaultStepFunction sf ^INDArray x ^INDArray line]
   (.step sf x line))
  ([^DefaultStepFunction sf ^INDArray parameters, ^INDArray search-direction, ^double step]
   (.step sf parameters search-direction step)))
