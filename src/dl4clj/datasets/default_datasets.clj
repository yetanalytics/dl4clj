(ns dl4clj.datasets.default-datasets
  (:import [org.deeplearning4j.datasets DataSets]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; build in datasets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-iris-ds
  "creates a new instance of the iris dataset"
  []
  (DataSets/iris))

(defn new-mnist-ds
  "creates a new instance of the mnist dataset"
  []
  (DataSets/mnist))
