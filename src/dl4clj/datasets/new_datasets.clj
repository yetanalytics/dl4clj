(ns dl4clj.datasets.new-datasets
  (:import [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset MultiDataSet])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-ds
  "Creates a DataSet object with the specified input and output.
  if they are not supplied, creates a new empty DataSet object

  :input (INDArray), the input to a model

  :output (INDArray), the targets/labels for the supplied input

  see: http://nd4j.org/doc/org/nd4j/linalg/dataset/DataSet.html"
  [& {:keys [input output]
      :as opts}]
  (if (contains-many? opts :input :output)
    (DataSet. input output)
    (DataSet.)))

(defn new-multi-ds
  "a dataset that contains multiple datasets

  see: http://nd4j.org/doc/org/nd4j/linalg/dataset/MultiDataSet.html"
  ;; come back and beef up this doc string
  ;; also ensure the arrays  are INDArrays of INDarrays
  ;; bc the constructor can accept these as just a single INDArray
  ;; make sure this is documented
  [& {:keys [features labels features-mask labels-mask]
      :as opts}]
  (cond (contains-many? opts :features :labels :features-mask :labels-mask)
        (MultiDataSet. features labels features-mask labels-mask)
        (contains-many? opts :features :labels)
        (MultiDataSet. features labels)
        :else
        (MultiDataSet.)))
