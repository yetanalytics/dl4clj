(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/dataset/api/DataSet.html"}
  nd4clj.linalg.dataset.api.data-set
  (:refer-clojure :exclude [get apply])
  (:import [org.nd4j.linalg.dataset.api DataSet]
           [org.nd4j.linalg.api.ndarray INDArray]
           [com.google.common.base Function]))

;;TODO
;; double check these fns
;; fill out doc strings

(defn add-feature-vector
  "With two arguments: Adds a feature for each example on to the current feature vector.
  With three arguments: The feature to add, and the example/row number"
  ([^DataSet this ^INDArray to-add]
   (.addFeatureVector this to-add))
  ([^DataSet this ^INDArray feature, example]
   (.addFeatureVector this feature (int example))))

(defn add-row
  ""
  [^DataSet this ^DataSet d, i]
  (.addRow this d (int i)))

;; (defn apply
;;   ""
;;   [^DataSet this ^Condition condition, ^Function function]
;;   (.apply this condition function))


(defn as-list [^DataSet this]
  ""
  (.asList this))

(defn batch-by
  "Partitions a dataset in to mini batches where each dataset in each list is of the specified number of examples"
  [^DataSet this num]
  (.batchBy this num))

(defn batch-by-num-labels [^DataSet this]
  (.batchByNumLabels this))

(defn binarize
  "Binarizes the dataset such that any number greater than cutoff is 1 otherwise zero"
  ([^DataSet this]
   (.binarize this))
  ([^DataSet this cutoff]
   (.binarize this (double cutoff))))

(defn copy
  "Clone the dataset"
  [^DataSet this]
  (.copy this))

(defn data-set-batches
  "Partitions the data transform by the specified number."
  [^DataSet this num]
  (.dataSetBatches this (int num)))

(defn divide-by
  ""
  [^DataSet this num]
  (.divideBy this (int num)))

(defn example-maxs
  ""
  [^DataSet this]
  (.exampleMaxs this))

(defn example-means
  ""
  [^DataSet this]
  (.exampleMeans this))

(defn example-sums
  ""
  [^DataSet this]
  (.exampleSums this))

(defn filter-and-strip
  "Strips the dataset down to the specified labels and remaps them"
  [^DataSet this labels]
  (.filterAndStrip this labels))

(defn filter-by
  "Strips the data transform of all but the passed in labels"
  [^DataSet this labels]
  (.filterBy this labels))

(defn get
  "Gets a copy of example i"
  [^DataSet this i & more]
  (.get this (int-array (conj more i))))

(defn get-column-names
  "Optional column names of the data transform, this is mainly used for interpeting what columns are in the dataset"
  [^DataSet this]
  (.getColumnNames this))

(defn get-feature-matrix
  "Get the feature matrix (inputs for the data)"
  [this]
  (.getFeatureMatrix this))

(defn get-features
  ""
  [^DataSet this]
  (.getFeatures this))

(defn get-label-names
  "Gets the optional label names"
  [^DataSet this]
  (.getLabelNames this))

(defn get-labels
  "Returns the labels for the dataset"
  [^DataSet this]
  (.getLabels this))

(defn iterator
  ""
  [^DataSet this]
  (.iterator this))

(defn multiply-by
  ""
  [^DataSet this num]
  (.multiplyBy this (double num)))

(defn normalize
  ""
  [^DataSet this]
  (.normalize this))

(defn normalize-zero-mean-zero-unit-variance
  "Subtract by the column means and divide by the standard deviation"
  [^DataSet this]
  (.normalizeZeroMeanZeroUnitVariance this))

(defn num-examples
  ""
  [^DataSet this]
  (.numExamples this))

(defn num-inputs
  "The number of inputs in the feature matrix"
  [^DataSet this]
  (.numInputs this))

(defn num-outcomes
  ""
  [^DataSet this]
  (.numOutcomes this))

(defn outcome
  ""
  [^DataSet this]
  (.outcome this))

(defn reshape
  "Reshapes the input in to the given rows and columns"
  [^DataSet this rows cols]
  (.reshape this (int rows) (int cols)))

(defn round-to-the-nearest
  ""
  [^DataSet this round-to]
  (.roundToTheNearest this (int round-to)))

(defn sample
  "Sample with/without replacement and a given/random rng"
  ([^DataSet this num-samples]
   (.sample this (int num-samples)))
  ([^DataSet this num-samples with-replacement?]
   (.sample this (int num-samples) (boolean with-replacement?)))
  ([^DataSet this num-samples with-replacement? rng]
   (.sample this (int num-samples) (boolean with-replacement?) rng)))

(defn save-dataset!
  ""
  [^DataSet this destination]
  (.save this (clojure.java.io/as-file destination)))

(defn scale
  "Divides the input data transform by the max number in each row"
  [^DataSet this]
  (.scale this))

(defn scale-min-and-max
  [^DataSet this]
  (.scaleMinAndMax this (double min) (double max)))

(defn set-column-names
  "Sets the column names, will throw an exception if the column names don't match the number of columns"
  [^DataSet this column-names]
  (.setColumnNames this column-names))

(defn set-features
  ""
  [^DataSet this ^INDArray features]
  (.setFeatures this features))

(defn set-label-names
  "Sets the label names, will throw an exception if the passed in label names doesn't equal the number of outcomes"
  [^DataSet this label-names]
  (.setLabelNames this label-names))

(defn set-labels
  ""
  [^DataSet this ^INDArray labels]
  (.setLabels this labels))

(defn set-new-number-of-labels
  "Clears the outcome matrix setting a new number of labels"
  [^DataSet this labels]
  (.setNewNumberOfLabels this labels))

(defn set-outcome
  "Sets the outpcome or a particular example"
  [^DataSet this example, label]
  (.setOutcome this (int example) (int label)))

(defn shuffle
  ""
  [^DataSet this]
  (.shuffle this))

(defn sort-and-batch-by-num-labels
  "Sorts the dataset by label: Splits the data transform such that examples are sorted by their labels."
  [^DataSet this]
  (.sortAndBatchByNumLabels this))

(defn sort-by-label
  "Organizes the dataset to minimize sampling error while still allowing efficient batching."
  [^DataSet this]
  (.sortByLabel this))

(defn split-test-and-train
  "Splits a dataset in to test and train"
  [^DataSet this num-holdout]
  (.splitTestAndTrain this (int num-holdout)))

(defn squish-to-range
  "Squeezes input data to a max and a min"
  [^DataSet this min, max]
  (.squishToRange this (double min) (double max)))

(defn validate
  ""
  [^DataSet this]
  (.validate this))
