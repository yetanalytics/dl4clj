(ns ^{:doc "see http://nd4j.org/doc/org/nd4j/linalg/dataset/DataSet.html"}
  nd4clj.linalg.dataset.data-set
  (:refer-clojure :exclude [get])
  (:import [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defn data-set
  ([] (DataSet.))
  ([^INDArray input ^INDArray output]
   (DataSet. input output)))
;; update http://nd4j.org/doc/index.html?org/nd4j/linalg/dataset/api/iterator/DataSetIterator.html
