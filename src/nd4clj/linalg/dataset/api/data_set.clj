(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/dataset/api/DataSet.html"}
  nd4clj.linalg.dataset.api.data-set
  (:refer-clojure :exclude [get])
  (:import [org.nd4j.linalg.dataset.api DataSet]
           [org.nd4j.linalg.api.ndarray INDArray]
           [com.google.common.base Function]))

(defmulti add-feature-vector
  (fn [^DataSet this ^INDArray to-add] (type this)))

(defmulti add-feature-vector
  (fn [^DataSet this ^INDArray feature, example] (type this))) 

(defmulti add-row
  (fn [^DataSet this ^DataSet d, i] (type this))) 

;; (defmulti apply
;;   (fn [^DataSet this ^Condition condition, ^Function function] (type this)))


(defmulti as-list
  ""
  (fn [^DataSet this] (type this))) 

(defmulti batch-by
  (fn [^DataSet this num] (type this))) 

(defmulti batch-by-num-labels
  ""
  (fn [^DataSet this] (type this))) 

(defmulti binarize
  ""
  (fn [^DataSet this & more] (type this))) 

(defmulti copy
  ""
  (fn [^DataSet this] (type this))) 

(defmulti data-set-batches
  (fn [^DataSet this num] (type this))) 

(defmulti divide-by
  (fn [^DataSet this num] (type this))) 

(defmulti example-maxs
  ""
  (fn [^DataSet this] (type this))) 

(defmulti example-means
  ""
  (fn [^DataSet this] (type this))) 

(defmulti example-sums
  ""
  (fn [^DataSet this] (type this))) 

(defmulti filter-and-strip
  (fn [^DataSet this labels] (type this))) 

(defmulti filter-by
  (fn [^DataSet this labels] (type this))) 

(defmulti get
  (fn [^DataSet this i & more] (type this))) 
 
(defmulti get-column-names
  ""
  (fn [^DataSet this] (type this))) 

(defmulti get-feature-matrix
  ""
  (fn [^DataSet this] (type this))) 

(defmulti get-features
  ""
  (fn [^DataSet this] (type this))) 

(defmulti get-label-names
  ""
  (fn [^DataSet this] (type this))) 

(defmulti get-labels
  ""
  (fn [^DataSet this] (type this))) 

(defmulti iterator
  ""
  (fn [^DataSet this] (type this))) 

(defmulti multiply-by
  (fn [^DataSet this ^double num] (type this))) 

(defmulti normalize
  ""
  (fn [^DataSet this] (type this))) 

(defmulti normalize-zero-mean-zero-unit-variance
  ""
  (fn [^DataSet this] (type this))) 

(defmulti num-examples
  ""
  (fn [^DataSet this] (type this))) 

(defmulti num-inputs
  ""
  (fn [^DataSet this] (type this))) 

(defmulti num-outcomes
  ""
  (fn [^DataSet this] (type this))) 

(defmulti outcome
  ""
  (fn [^DataSet this] (type this))) 

(defmulti reshape
  (fn [^DataSet this rows cols] (type this))) 

(defmulti round-to-the-nearest
  (fn [^DataSet this round-to] (type this))) 

(defmulti sample
  (fn [^DataSet this num-samples & more] (type this))) 

(defmulti scale
  ""
  (fn [^DataSet this] (type this))) 

(defmulti set-column-names
  (fn [^DataSet this column-names] (type this))) 

(defmulti set-features
  (fn [^DataSet this ^INDArray features] (type this))) 

(defmulti set-label-names
  (fn [^DataSet this label-names] (type this))) 

(defmulti set-labels
  (fn [^DataSet this ^INDArray labels] (type this))) 

(defmulti set-new-number-of-labels
  (fn [^DataSet this labels] (type this))) 

(defmulti set-outcome
  (fn [^DataSet this example, label] (type this))) 

(defmulti shuffle
  ""
  (fn [^DataSet this] (type this))) 

(defmulti sort-and-batch-by-num-labels
  ""
  (fn [^DataSet this] (type this))) 

(defmulti sort-by-label
  ""
  (fn [^DataSet this] (type this))) 

(defmulti split-test-and-train
  (fn [^DataSet this num-holdout] (type this))) 

(defmulti squish-to-range
  (fn [^DataSet this min, max] (type this))) 

(defmulti validate
  ""
  (fn [^DataSet this] (type this))) 


