(ns ^{:doc "see http://nd4j.org/doc/org/nd4j/linalg/dataset/DataSet.html"}
  nd4clj.linalg.dataset.data-set
  (:import [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.api.ndarray INDArray])
  (:require [dl4clj.utils :refer [contains-many?]]))

;; TODO
;; expand this ns

(defn data-set
  "Creates a DataSet object with the specified input and output.
  if they are not supplied, creates a new empty DataSet object

  :input (INDArray), the input to a model

  :output (INDArray), the targets/labels for the supplied input"
  [& {:keys [input output]
      :as opts}]
  (if (contains-many? opts :input :output)
    (DataSet. input output)
    (DataSet.)))
