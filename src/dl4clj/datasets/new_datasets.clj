(ns dl4clj.datasets.new-datasets
  (:import [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset MultiDataSet])
  (:require [dl4clj.utils :refer [contains-many? obj-or-code?]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn new-ds
  "Creates a DataSet object with the specified input and output.
  if they are not supplied, creates a new empty DataSet object

  :input (vec, matrix or INDArray), the input to a model

  :output (vec, matrix or INDArray), the targets/labels for the supplied input

  :as-code? (boolean), return the dl4j obj or the code for creating it

  see: http://nd4j.org/doc/org/nd4j/linalg/dataset/DataSet.html"
  [& {:keys [input output as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (if (contains-many? opts :input :output)
               `(DataSet. (vec-or-matrix->indarray ~input) (vec-or-matrix->indarray ~output))
               `(DataSet.))]
    (obj-or-code? as-code? code)))

(defn new-multi-ds
  "a dataset that contains multiple datasets

  see: http://nd4j.org/doc/org/nd4j/linalg/dataset/MultiDataSet.html"
  ;; come back and beef up this doc string
  ;; also ensure the arrays  are INDArrays of INDarrays
  ;; bc the constructor can accept these as just a single INDArray
  ;; make sure this is documented
  [& {:keys [features labels features-mask labels-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (let [f `(vec-or-matrix->indarray ~features)
        l `(vec-or-matrix->indarray ~labels)
        code (match [opts]
                    [{:features _ :labels _ :features-mask _ :labels-mask _}]
                    `(MultiDataSet. ~f ~l (vec-or-matrix->indarray ~features-mask)
                                    (vec-or-matrix->indarray ~labels-mask))
                    [{:features _ :labels _}]
                    `(MultiDataSet. ~f ~l)
                    :else
                    `(MultiDataSet.))]
    (obj-or-code? as-code? code)))
