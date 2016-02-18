(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/dataset/DataSet.html"}
  nd4clj.linalg.dataset.data-set
  (:refer-clojure :exclude [get])
  (:require [nd4clj.linalg.dataset.api.data-set :refer :all])
  (:import [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.api.ndarray INDArray]))


(defmethod add-feature-vector DataSet
  ([^DataSet this ^INDArray to-add]
   (.addFeatureVector this to-add))
  ([^DataSet this ^INDArray feature, example]
   (.addFeatureVector this ^INDArray feature, example)))

(defmethod add-row DataSet
  [^DataSet this ^DataSet d i]
  (.addRow this d (int i)))


(defmethod as-list DataSet
  [^DataSet this]
  (type this))

(defmethod batch-by DataSet
  [^DataSet this num]
  (.batchBy this (int num)))

(defmethod batch-by-num-labels DataSet
  [^DataSet this]
  (.batchByNumLabels this))

(defmethod binarize DataSet
  ([^DataSet this]
   (.binarize this))
  ([^DataSet this  cutoff]
   (.binarize this (double cutoff))))

(defmethod copy DataSet
  [^DataSet this]
  (.copy this))

(defmethod data-set-batches DataSet
  [^DataSet this num]
  (.dataSetBatches this (int num)))

(defmethod divide-by DataSet
  [^DataSet this num]
  (.divideBy this (int num))) 

(defmethod example-maxs DataSet
  [^DataSet this]
  (.exampleMaxs this))

(defmethod example-means DataSet
  [^DataSet this]
  (.exampleMeans this))

(defmethod example-sums DataSet
  [^DataSet this]
  (.exampleSums this))

(defmethod filter-and-strip DataSet
  [^DataSet this labels]
  (.filterAndStrip this (int-array labels))) 

(defmethod filter-by DataSet
  [^DataSet this labels]
  (.filterBy this (int-array labels))) 

(defmethod get DataSet
  ([^DataSet this i]
   (.get this (int i)))
  ([^DataSet this i & more]
   (.get this (int-array (conj more i)))))

(defmethod get-column-names DataSet
  [^DataSet this]
  (.getColumnNames this))

(defmethod get-feature-matrix DataSet
  [^DataSet this]
  (.getFeatureMatrix this))

(defmethod get-features DataSet
  [^DataSet this]
  (.getFeatures this))

(defmethod get-label-names DataSet
  [^DataSet this]
  (.getLabelNames this))

(defmethod get-labels DataSet
  [^DataSet this]
  (.getLabels this))

(defmethod iterator DataSet
  [^DataSet this]
  (.iterator this))

(defmethod multiply-by DataSet
  [^DataSet this num]
  (.multiplyBy this (double num))) 

(defmethod normalize DataSet
  [^DataSet this]
  (.normalize this))

(defmethod normalize-zero-mean-zero-unit-variance DataSet
  [^DataSet this]
  (.normalizeZeroMeanZeroUnitVariance this))

(defmethod num-examples DataSet
  [^DataSet this]
  (.numExamples this))

(defmethod num-inputs DataSet
  [^DataSet this]
  (.numInputs this))

(defmethod num-outcomes DataSet
  [^DataSet this]
  (.numOutcomes this))

(defmethod outcome DataSet
  [^DataSet this]
  (.outcome this))

(defmethod reshape DataSet
  [^DataSet this rows cols]
  (.reshape this rows cols))

(defmethod round-to-the-nearest DataSet
  [^DataSet this round-to]
  (.roundToTheNearest this (int round-to))) 

(defmethod sample DataSet
  ([^DataSet this num-samples]
   (.sample this (int num-samples)))
  ([^DataSet this num-samples with-replacement?]
   (.sample this (int num-samples) (boolean with-replacement?))))

(defmethod scale DataSet
  [^DataSet this]
  (.scale this))

(defmethod set-column-names DataSet
  [^DataSet this column-names]
  (.setColumnNames this column-names)) 

(defmethod set-features DataSet
  [^DataSet this ^INDArray features]
  (.setFeatures this features)) 

(defmethod set-label-names DataSet
  [^DataSet this label-names]
  (.setLabelNames this label-names)) 

(defmethod set-labels DataSet
  [^DataSet this ^INDArray labels]
  (.setLabels this ^INDArray labels)) 

(defmethod set-new-number-of-labels DataSet
  [^DataSet this labels]
  (.setNewNumberOfLabels this labels)) 

(defmethod set-outcome DataSet
  [^DataSet this example, label]
  (.setOutcome this (int example) (int label))) 

(defmethod shuffle DataSet
  [^DataSet this]
  (.shuffle this))

(defmethod sort-and-batch-by-num-labels DataSet
  [^DataSet this]
  (.sortAndBatchByNumLabels this))

(defmethod sort-by-label DataSet
  [^DataSet this]
  (.sortByLabel this))

(defmethod split-test-and-train DataSet
  [^DataSet this num-holdout]
  (.splitTestAndTrain this (int num-holdout)))

(defmethod squish-to-range DataSet
  [^DataSet this min, max]
  (.squishToRange this (double min) (double max))) 

(defmethod validate DataSet
  [^DataSet this]
  (.validate this))

(defn data-set 
  ([] (DataSet.))
  ([^INDArray input ^INDArray output]
   (DataSet. input output)))
