(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/optimize/stepfunctions/DefaultStepFunction.html"}
  dl4clj.optimize.step-functions.default-step-function
  (:import [org.deeplearning4j.optimize.stepfunctions DefaultStepFunction]))

;; revisit this ns
;; was from og author, make step keyword args
;; see if there are additional methods to implement


(defn default-step-function []
  (DefaultStepFunction.))

#_(defn step
  ([sf]
   (.step sf))
  ([sf x line]
   (.step sf x line))
  ([sf parameters search-direction step]
   (.step sf parameters search-direction step)))
