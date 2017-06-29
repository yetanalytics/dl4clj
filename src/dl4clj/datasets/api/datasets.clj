(ns dl4clj.datasets.api.datasets
  (:import [org.nd4j.linalg.dataset.api DataSet]
           [org.nd4j.linalg.api.ndarray INDArray]
           [java.util Random])
  (:require [dl4clj.utils :refer [array-of contains-many?]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn add-feature-vector!
  "adds a feature vector to a dataset, if :to-add is supplied, the fn adds a
   feature for each example on to the current feature vector.  Otherwise,
   this fn expects :feature and :example-idx.  Those args will add the feature
   to the example at the given idx

   :feature (INDArray or vector), the feature vector to add

   :example-idx (int), index of the example you want to add the feature vector to

   :to-add (INDArray or vector), need to test to clarify

  returns the supplied dataset"
  [& {:keys [ds feature example-idx to-add]
      :as opts}]
  (let [f (vec-or-matrix->indarray features)
        ta (vec-or-matrix->indarray to-add)]
   (if (contains? opts :to-add)
        (doto ds (.addFeatureVector ta))
        (doto ds (.addFeatureVector f example-idx)))))

(defn add-row!
  "adds a dataset object to an existing datset object as a new row

  :row (DataSet), the datset to add as a row

  :idx (int), the index at which to add the row

  returns the dataset"
  [& {:keys [ds row idx]}]
  (doto ds (.addRow row idx)))

(defn as-list
  "Extract each example in the DataSet into its own DataSet object,
  and return all of them as a list"
  [ds]
  (.asList ds))

(defn batch-by!
  "Partitions a dataset in to mini batches where each dataset in each list
  is of the specified number of examples

  :n-examples (int), the desired number of examples within each datset"
  [& {:keys [ds n-examples]}]
  (.batchBy ds n-examples))

(defn batch-by-n-labels!
  "partitions a dataset in to mini batches where each datset in each list
   has n-labels examples in it"
  [ds]
  (.batchByNumLabels ds))

(defn binarize!
  "Binarizes the dataset such that any number greater than :cutoff is 1 otherwise zero"
  [& {:keys [ds cutoff]
      :as opts}]
  (if (contains? opts :cutoff)
    (doto ds (.binarize cutoff))
    (doto ds .binarize)))

(defn divide-by!
  "divide the features in a datset by a scalar"
  [& {:keys [ds scalar]}]
  (doto ds (.divideBy scalar)))

(defn get-eample-maxs
  "returns the max from each example (i believe)"
  [ds]
  (.exampleMaxs ds))

(defn get-example-means
  "returns the means from each example (i believe)"
  [ds]
  (.exampleMeans ds))

(defn get-example-sums
  "returns the sums of each example (i believe)"
  [ds]
  (.exampleSums ds))

(defn filt-and-strip-labels!
  "Strips the dataset down to the specified labels (by indexs) and remaps them

  :label-idxs (coll), a collection of label indexes"
  [& {:keys [ds label-idxs]}]
  (doto ds (.filterAndStrip (int-array label-idxs))))

(defn filter-by!
  "Strips the data set of all but the passed in labels

  :label-idxs (coll), a collection of albel indexes"
  [& {:keys [ds label-idxs]}]
  (.filterBy ds (int-array label-idxs)))

(defn get-example
  "returns a specified example(s) from a dataset

  :idx (coll or int), the index(es) you want to get out of the dataset"
  [& {:keys [ds idx]}]
  (if (int? idx)
    (.get ds idx)
    (.get ds (int-array idx))))

(defn get-column-names
  "return the names of the columns in a dataset"
  [ds]
  (.getColumnNames ds))

(defn get-features
  "returns the features array for the dataset"
  [ds]
  (.getFeatures ds))

(defn get-features-mask-array
  "Input mask array: a mask array for input, where each value is in {0,1} in order
  to specify whether an input is actually present or not."
  [ds]
  (.getFeaturesMaskArray ds))

(defn get-label-names
  "return the label for a single index or for a collection of indexes

  :idx (int, vec or INDArray), the idex(es) of labels you want the name(s) of

  :as-list? (boolean), do you want all of the label names as a list?

  if only :ds is supplied, all labels will be returned as an INDArray"
  [& {:keys [ds idx as-list?]
      :or {all? false}}]
  (cond
    (true? as-list?) (.getLabelNamesList ds)
    (int? idx) (.getLabelName ds idx)
    (coll? idx) (.getLabelNames ds (vec-or-matrix->indarray idx))
    :else
    (.getLabels ds)))

(defn get-labels-mask-array
  "Labels (output) mask array: a mask array for input, where each value is in {0,1}
  in order to specify whether an output is actually present or not."
  [ds]
  (.getLabelsMaskArray ds))

(defn get-range
  "returns a dataset containing the range of examples from the orginal dataset"
  [& {:keys [ds from to]}]
  (.getRange ds from to))

(defn has-mask-arrays?
  "does this dataset contain any mask arrays?"
  [ds]
  (.hasMaskArrays ds))

(defn get-ds-id
  "returns the id of the dataset"
  [ds]
  (.id ds))

(defn new-ds-iter
  "creates an iterator for the supplied dataset"
  [ds]
  (.iterator ds))

(defn get-label-counts
  "Calculate and return a count of each label, by index."
  [ds]
  (.labelCounts ds))

(defn load-ds!
  "loads a dataset from a given input stream or a file path

  :in (InputStream), a source to load a dataset from

  :file-path (str), the path to a file containing the dataset"
  [& {:keys [file-path in]
      :as opts}]
  (if (contains? opts :in)
    (.load in)
    (.load (clojure.java.io/as-file file-path))))

(defn multiply-by!
  "multiply the features in a dataset by a scalar

  the scalar should be a double"
  [& {:keys [ds scalar]}]
  (doto ds (.multiplyBy scalar)))

(defn normalize!
  "normalize the dataset to have a mean of 0 and a stdev of 1 per input"
  [ds]
  (doto ds .normalize))

(defn num-examples
  "returns the number of examples in the dataset"
  [ds]
  (.numExamples ds))

(defn num-inputs
  "the number of input values
   - the size of the features INDArray per example"
  [ds]
  (.numInputs ds))

(defn num-outcomes
  "returns the number of outcomes
   - size of the labels array for each example"
  [ds]
  (.numOutcomes ds))

(defn get-outcome
  "returns the size of the outcome of the current example"
  [ds]
  (.outcome ds))

(defn reshape!
  "reshapes a datset to have the desired number of rows and columns"
  [& {:keys [ds rows cols]}]
  (.reshape ds rows cols))

(defn round-to-the-nearest!
  "rounts values in the dataset to the supplied nearest value

  :round-to (int), the value you want things rounded to"
  [& {:keys [ds round-to]}]
  (doto ds (.roundToTheNearest round-to)))

(defn sample-ds
  "Sample with/without replacement and a given/random rng"
  [& {:keys [ds n-samples with-replacement? seed]
      :as opts}]
  ;; this is the wrong type of random
  ;; test to see if it will still work with util.Random, kinda doubt it
  (let [rng (new Random seed)]
   (cond (contains-many? opts :n-samples :with-replacement? :seed)
         (.sample ds n-samples rng with-replacement?)
         (contains-many? opts :n-samples :seed)
         (.sample ds n-samples rng)
         (contains-many? opts :n-samples :with-replacement?)
         (.sample ds n-samples with-replacement?)
         :else
         (.sample ds n-samples))))

(defn save-ds!
  "saves a datset to a given file or output stream

  :out (OutputStream), an output stream to save the dataset to

  :file-path (str), a string to a file you want to save the dataset in"
  [& {:keys [ds file-path out]
      :as opts}]
  (if (contains? opts :out)
    (doto ds (.save out))
    (doto ds (.save (clojure.java.io/as-file file-path)))))

(defn scale-ds!
  "scales a the input data to be in the range of :max-val and :min-val if supplied.

  otherwise divides the input data by the max value in each row"
  [& {:keys [ds max-val min-val]
      :as opts}]
  (if (contains-many? opts :max-val :min-val)
    (doto ds (.scaleMinAndMax min-val max-val))
    (doto ds (.scale))))

(defn set-column-names!
  "sets the column names for the dataset and returns the dataset

  :names (list), a list of strings"
  [& {:keys [ds names]}]
  (doto ds (.setColumnNames names)))

(defn set-features!
  "set the features array for the dataset

  :features (vec or matrix), the data you want to set as the features"
  [& {:keys [ds features]}]
  (doto ds (.setFeatures (vec-or-matrix->indarray features))))

(defn set-features-mask-array!
  "set the features mask array for the supplied dataset

  :input-mask (vec or matrix), the mask to be set"
  [& {:keys [ds input-mask]}]
  (doto ds (.setFeaturesMaskArray (vec-or-matrix->indarray input-mask))))

(defn set-label-names!
  "sets the label names

  :label-names (list), a list of label names you want assigned"
  [& {:keys [ds label-names]}]
  (doto ds (.setLabelNames label-names)))

(defn set-labels!
  "sets the labels for a dataset

  :labels (vec or matrix), the values for the labels"
  [& {:keys [ds labels]}]
  (doto ds (.setLabels (vec-or-matrix->indarray labels))))

(defn set-labels-mask-array!
  "sets the labels mask array for the dataset

  :mask-array (vec or matrix), the mask array to set"
  [& {:keys [ds mask-array]}]
  (doto ds (.setLabelsMaskArray (vec-or-matrix->indarray mask-array))))

(defn set-new-number-of-labels!
  "sets a new number of labels for the dataset"
  [& {:keys [ds n-labels]}]
  (doto ds (.setNewNumberOfLabels n-labels)))

(defn set-outcome!
  "sets an outcome for a given example"
  [& {:keys [ds example-idx label-idx]}]
  (doto ds (.setOutcome example-idx label-idx)))

(defn shuffle-ds!
  [ds]
  (doto ds .shuffle))

(defn sort-and-batch-by-num-labels!
  "sorts a dataset by the labels and then batches by the number of labels"
  [ds]
  (.sortAndBatchByNumLabels ds))

(defn sort-by-label!
  "sorts a dataset by its labels"
  [ds]
  (doto ds .sortByLabel))

(defn split-test-and-train!
  "split the dataset into two datasets randomly"
  [& {:keys [ds percent-train n-holdout seed]
      :as opts}]
  (cond (contains-many? opts :n-holdout :seed)
        (.splitTestAndTrain ds n-holdout (new Random seed))
        (contains? opts :n-holdout)
        (.splitTestAndTrain ds n-holdout)
        (contains? opts :percent-train)
        (.splitTestAndTrain ds percent-train)))

(defn squish-to-range!
  "Squeezes input data to a max and a min"
  [& {:keys [ds min-val max-val]}]
  (doto ds (.squishToRange min-val max-val)))

(defn validate-ds!
  "validates a dataset"
  [ds]
  (doto ds (.validate)))
