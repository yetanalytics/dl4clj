(ns dl4clj.datasets.default-datasets
  (:import [org.deeplearning4j.datasets DataSets]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; build in datasets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-iris-ds
  "creates a new instance of the iris dataset"
  [& {:keys [as-code?]
      :or {as-code? false}}]
  (if as-code?
    `(DataSets/iris)
    (DataSets/iris)))

(defn new-mnist-ds
  "creates a new instance of the mnist dataset"
  [& {:keys [as-code?]
      :or {as-code? false}}]
  (if as-code?
    `(DataSets/mnist)
    (DataSets/mnist)))
