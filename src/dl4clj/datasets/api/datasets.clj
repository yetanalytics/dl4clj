(ns dl4clj.datasets.api.datasets
  (:import [org.nd4j.linalg.dataset.api DataSet]
           [org.nd4j.linalg.api.ndarray INDArray]
           [java.util Random])
  (:require [dl4clj.utils :refer [array-of contains-many? obj-or-code? eval-if-code]]
            [clojure.core.match :refer [match]]
            [dl4clj.indarray :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-eample-maxs
  "returns the max from each example"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.exampleMaxs ~ds))
         :else
         (.exampleMaxs ds)))

(defn get-example-means
  "returns the means from each example"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.exampleMeans ~ds))
         :else
         (.exampleMeans ds)))

(defn get-example-sums
  "returns the sums of each example"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.exampleSums ~ds))
         :else
         (.exampleSums ds)))

(defn get-example
  "returns a specified example(s) from a dataset

  :idx (coll or int), the index(es) you want to get out of the dataset"
  [& {:keys [ds idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :idx (:or (_ :guard number?)
                     (_ :guard seq?))}]
         (obj-or-code? as-code? `(.get ~ds (int ~idx)))
         [{:ds (_ :guard seq?)
           :idx (:or (_ :guard coll?)
                     (_ :guard seq?))}]
         (obj-or-code? as-code? `(.get ~ds (int-array ~idx)))
         [{:ds _
           :idx (_ :guard number?)}]
         (.get ds idx)
         [{:ds _
           :idx (_ :guard coll?)}]
         (.get ds (int-array idx))
         :else
         (let [[ds-obj idx-n] (eval-if-code [ds seq?]
                                            [idx seq?])]
           (if (number? idx-n)
             (.get ds-obj idx-n)
             (.get ds-obj (int-array idx-n))))))

(defn get-column-names
  "return the names of the columns in a dataset"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getColumnNames ~ds))
         :else
         (.getColumnNames ds)))

(defn get-features
  "returns the features array for the dataset"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getFeatures ~ds))
         :else
         (.getFeatures ds)))

(defn get-features-mask-array
  "Input mask array: a mask array for input, where each value is in {0,1} in order
  to specify whether an input is actually present or not."
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getFeaturesMaskArray ~ds))
         :else
         (.getFeaturesMaskArray ds)))

(defn get-label-names
  "return the label for a single index or for a collection of indexes

  :idx (int, vec or INDArray), the idex(es) of labels you want the name(s) of

  :as-list? (boolean), do you want all of the label names as a list?

  if only :ds is supplied, all labels will be returned as an INDArray"
  [& {:keys [ds idx as-list? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :idx (:or (_ :guard number?)
                     (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getLabelName ~ds (int ~idx)))
         [{:ds (_ :guard seq?)
           :idx (:or (_ :guard vector?)
                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.getLabelNames ~ds (vec-or-matrix->indarray ~idx)))
         [{:ds _ :idx (_ :guard number?)}]
         (.getLabelName ds idx)
         [{:ds _ :idx (_ :guard vector?)}]
         (.getLabelNames ds (vec-or-matrix->indarray idx))
         ;; if it comes in as an existing INDArray
         [{:ds _ :idx _}]
         (let [[idx-n] (eval-if-code [idx seq?])]
           (if (vector? idx-n)
             (.getLabelNames ds (vec-or-matrix->indarray idx-n))
             (.getLabelNames ds idx-n)))
         [{:ds (_ :guard seq?)
           :as-list? true}]
         (obj-or-code? as-code? `(.getLabelNamesList ~ds))
         [{:ds _ :as-list? true}]
         (.getLabelNamesList ds)
         [{:ds (_ :guard seq?)}]
         (obj-or-code? as-code? `(.getLabels ~ds))
         :else
         (.getLabels ds)))

(defn get-labels-mask-array
  "Labels (output) mask array: a mask array for input, where each value is in {0,1}
  in order to specify whether an output is actually present or not."
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLabelsMaskArray ~ds))
         :else
         (.getLabelsMaskArray ds)))

(defn get-range
  "returns a dataset containing the range of examples from the orginal dataset"
  [& {:keys [ds from to as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :from (:or (_ :guard number?)
                      (_ :guard seq?))
           :to (:or (_ :guard number?)
                    (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getRange ~ds (int ~from) (int ~to)))
         :else
         (let [[ds-obj from-n to-n] (eval-if-code [ds seq?] [from seq? number?]
                                                  [to seq? number?])]
           (.getRange ds-obj from-n to-n))))

(defn get-ds-id
  "returns the id of the dataset"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.id ~ds))
         :else
         (.id ds)))

(defn get-label-counts
  "Calculate and return a count of each label, by index."
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.labelCounts ~ds))
         :else
         (.labelCounts ds)))

(defn num-examples
  "returns the number of examples in the dataset"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.numExamples ~ds))
         :else
         (.numExamples ds)))

(defn num-inputs
  "the number of input values
   - the size of the features INDArray per example"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.numInputs ~ds))
         :else
         (.numInputs ds)))

(defn num-outcomes
  "returns the number of outcomes
   - size of the labels array for each example"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.numOutcomes ~ds))
         :else
         (.numOutcomes ds)))

(defn get-outcome
  "returns the size of the outcome of the current example"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.outcome ~ds))
         :else
         (.outcome ds)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn set-column-names!
  "sets the column names for the dataset and returns the dataset

  :names (list), a list of strings"
  [& {:keys [ds names as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :names (_ :guard list?)}]
         (obj-or-code? as-code? `(doto ~ds (.setColumnNames ~names)))
         :else
         ;; consider guarding against vectors
         (doto ds (.setColumnNames names))))

(defn set-features!
  "set the features array for the dataset

  :features (vec or INDarray), the data you want to set as the features"
  [& {:keys [ds features as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.setFeatures (vec-or-matrix->indarray ~features))))
         :else
         (let [[obj-ds features-vec] (eval-if-code [ds seq?]
                                                   [features seq?])]
           (doto obj-ds (.setFeatures (vec-or-matrix->indarray features-vec))))))

(defn set-features-mask-array!
  "set the features mask array for the supplied dataset

  :input-mask (vec or INDarray), the mask to be set"
  [& {:keys [ds input-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :input-mask (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.setFeaturesMaskArray (vec-or-matrix->indarray ~input-mask))))
         :else
         (let [[obj-ds i-mask-vec] (eval-if-code [ds seq?] [input-mask seq?])]
           (doto obj-ds (.setFeaturesMaskArray (vec-or-matrix->indarray i-mask-vec))))))

(defn set-label-names!
  "sets the label names

  :label-names (list), a list of label names you want assigned"
  [& {:keys [ds label-names as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :label-names (_ :guard list?)}]
         (obj-or-code? as-code? `(doto ~ds (.setLabelNames ~label-names)))
         :else
         ;; consider guarding against vectors for label-names
         (doto ds (.setLabelNames label-names))))

(defn set-labels!
  "sets the labels for a dataset

  :labels (vec or INDarray), the values for the labels"
  [& {:keys [ds labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.setLabels (vec-or-matrix->indarray ~labels))))
         :else
         (let [[ds-obj labels-vec] (eval-if-code [ds seq?] [labels seq? vector?])]
          (doto ds-obj (.setLabels (vec-or-matrix->indarray labels-vec))))))

(defn set-labels-mask-array!
  "sets the labels mask array for the dataset

  :mask-array (vec or INDarray), the mask array to set"
  [& {:keys [ds mask-array as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.setLabelsMaskArray (vec-or-matrix->indarray ~mask-array))))
         :else
         (let [[ds-obj m-array] (eval-if-code [ds seq?] [mask-array seq?])]
           (doto ds-obj (.setLabelsMaskArray (vec-or-matrix->indarray m-array))))))

(defn set-new-number-of-labels!
  "sets a new number of labels for the dataset"
  [& {:keys [ds n-labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :n-labels (:or (_ :guard number?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.setNewNumberOfLabels (int ~n-labels))))
         :else
         (let [[ds-obj num-labels] (eval-if-code [ds seq?]
                                                 [n-labels seq? number?])]
           (doto ds-obj (.setNewNumberOfLabels (int num-labels))))))

(defn set-outcome!
  "sets an outcome for a given example"
  [& {:keys [ds example-idx label-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :example-idx (:or (_ :guard number?)
                             (_ :guard seq?))
           :label-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.setOutcome (int ~example-idx) (int ~label-idx))))
         :else
         (let [[ds-obj example-idx-n label-idx-n]
               (eval-if-code [ds seq?] [example-idx seq? number?]
                             [label-idx seq? number?])]
           (doto ds-obj (.setOutcome example-idx-n label-idx-n)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; manipulation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn add-feature-vector!
  "adds a feature vector to a dataset, if :to-add is supplied, the fn adds a
   feature for each example on to the current feature vector.  Otherwise,
   this fn expects :feature and :example-idx.  Those args will add the feature
   to the example at the given idx

   :feature (INDArray or vector), the feature vector to add

   :example-idx (int), index of the example you want to add the feature vector to

   :to-add (INDArray or vector), need to test to clarify

  returns the supplied dataset"
  [& {:keys [ds feature example-idx to-add as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts as-code?)]
         [{:ds (_ :guard seq?)
           :feature (:or (_ :guard vector?)
                         (_ :guard seq?))
           :example-idx (:or (_ :guard number?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.addFeatureVector (vec-or-matrix->indarray ~feature)
                                       (int ~example-idx))))
         [{:ds _ :feature _ :example-idx _}]
         (let [[ds-obj feature-vec idx-n]
               (eval-if-code [ds seq?] [feature seq?] [example-idx seq? number?])]
           (doto ds-obj (.addFeatureVector (vec-or-matrix->indarray feature-vec)
                                           idx-n)))
         [{:ds (_ :guard seq?)
           :to-add (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.addFeatureVector (vec-or-matrix->indarray ~to-add))))
         [{:ds _ :to-add _}]
         (let [[ds-obj to-add-vec] (eval-if-code [ds seq?] [to-add seq?])]
           (doto ds-obj (.addFeatureVector (vec-or-matrix->indarray to-add-vec))))))

(defn add-row!
  "adds a dataset object to an existing datset object as a new row

  :row (DataSet), the datset to add as a row

  :idx (int), the index at which to add the row

  returns the dataset"
  [& {:keys [ds row idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :row (_ :guard seq?)
           :idx (:or (_ :guard number?)
                     (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.addRow ~row ~idx)))
         :else
         (let [[ds-obj row-obj idx-n] (eval-if-code [ds seq?] [row seq?]
                                                    [idx seq? number?])]
           (doto ds-obj (.addRow row-obj idx-n)))))

(defn batch-by!
  "Partitions a dataset in to mini batches where each dataset in each list
  is of the specified number of examples

  :n-examples (int), the desired number of examples within each datset"
  [& {:keys [ds n-examples as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :n-examples (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.batchBy ~ds (int ~n-examples)))
         :else
         (let [[n-e] (eval-if-code [n-examples seq? number?])]
           (.batchBy ds (int n-e)))))

(defn batch-by-n-labels!
  "partitions a dataset in to mini batches where each datset in each list
   has n-labels examples in it"
  [ds & {:keys [as-code?]
         :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.batchByNumLabels ~ds))
         :else
         (.batchByNumLabels ds)))

(defn binarize!
  "Binarizes the dataset such that any number greater than :cutoff is 1 otherwise zero"
  [& {:keys [ds cutoff as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :cutoff (:or (_ :guard number?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.binarize (double ~cutoff))))
         [{:ds _ :cutoff _}]
         (let [[cutoff-n] (eval-if-code [cutoff seq? number?])]
           (doto ds (.binarize cutoff-n)))
         [{:ds (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~ds .binarize))
         :else
         (doto ds .binarize)))

(defn divide-by!
  "divide the features in a datset by a scalar"
  [& {:keys [ds scalar as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :scalar (:or (_ :guard number?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.divideBy (int ~scalar))))
         :else
         (let [[scalar-n] (eval-if-code [scalar seq? number?])]
           (doto ds (.divideBy scalar-n)))))

(defn filt-and-strip-labels!
  "Strips the dataset down to the specified labels (by indexs) and remaps them

  :label-idxs (coll), a collection of label indexes"
  [& {:keys [ds label-idxs as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :label-idxs (:or (_ :guard coll?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.filterAndStrip (int-array ~label-idxs))))
         :else
         (doto ds (.filterAndStrip (int-array label-idxs)))))

(defn filter-by!
  "Strips the data set of all but the passed in labels

  :label-idxs (coll), a collection of albel indexes"
  [& {:keys [ds label-idxs as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :label-idxs (:or (_ :guard coll?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.filterBy ~ds (int-array ~label-idxs)))
         :else
         (let [[idx-coll] (eval-if-code [label-idxs seq? coll?])]
           (.filterBy ds (int-array idx-coll)))))

(defn multiply-by!
  "multiply the features in a dataset by a scalar

  the scalar should be a double"
  [& {:keys [ds scalar as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts as-code?)]
         [{:ds (_ :guard seq?)
           :scalar (:or (_ :guard number?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.multiplyBy (double ~scalar))))
         :else
         (let [[scalar-n] (eval-if-code [scalar seq? number?])]
           (doto ds (.multiplyBy scalar-n)))))

(defn normalize!
  "normalize the dataset to have a mean of 0 and a stdev of 1 per input"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~ds .normalize))
         :else
         (doto ds .normalize)))

(defn reshape!
  "reshapes a datset to have the desired number of rows and columns"
  [& {:keys [ds rows cols as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :rows (:or (_ :guard number?)
                      (_ :guard seq?))
           :cols (:or (_ :guard number?)
                      (_ :guard seq?))}]
         (obj-or-code? as-code? `(.reshape ~ds ~rows ~cols))
         :else
         (let [[row-n col-n] (eval-if-code [rows seq? number?]
                                           [cols seq? number?])]
           (.reshape ds row-n col-n))))

(defn round-to-the-nearest!
  "rounts values in the dataset to the supplied nearest value

  :round-to (int), the value you want things rounded to"
  [& {:keys [ds round-to as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :round-to (:or (_ :guard number?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.roundToTheNearest ~round-to)))
         :else
         (let [[round-to-n] (eval-if-code [round-to seq? number?])]
           (doto ds (.roundToTheNearest round-to-n)))))

(defn scale-ds!
  "scales a the input data to be in the range of :max-val and :min-val if supplied.

  otherwise divides the input data by the max value in each row"
  [& {:keys [ds max-val min-val as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :max-val (:or (_ :guard number?)
                         (_ :guard seq?))
           :min-val (:or (_ :guard number?)
                         (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~ds (.scaleMinAndMax (double ~min-val)
                                                            (double ~max-val))))
         [{:ds _
           :max-val _
           :min-val _}]
         (let [[max-n min-n] (eval-if-code [max-val seq? number?]
                                           [min-val seq? number?])]
           (doto ds (.scaleMinAndMax min-n max-n)))
         [{:ds (_ :guard seq?)}]
         `(doto ~ds .scale)
         [{:ds _}]
         (doto ds .scale)))

(defn shuffle-ds!
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~ds .shuffle))
         :else
         (doto ds .shuffle)))

(defn sort-and-batch-by-num-labels!
  "sorts a dataset by the labels and then batches by the number of labels"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.sortAndBatchByNumLabels ~ds))
         :else
         (.sortAndBatchByNumLabels ds)))

(defn sort-by-label!
  "sorts a dataset by its labels"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~ds .sortByLabel))
         :else
         (doto ds .sortByLabel)))

(defn split-test-and-train!
  "split the dataset into two datasets randomly"
  [& {:keys [ds percent-train n-holdout seed as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :n-holdout (:or (_ :guard number?)
                           (_ :guard seq?))
           :seed (:or (_ :guard number?)
                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.splitTestAndTrain ~ds (int ~n-holdout) (new Random ~seed)))
         [{:ds _ :n-holdout _ :seed _}]
         (let [[holdout-n seed-n] (eval-if-code [n-holdout seq? number?]
                                                [seed seq? number?])]
           (.splitTestAndTrain ds holdout-n (new Random seed-n)))
         [{:ds (_ :guard seq?)
           :n-holdout (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.splitTestAndTrain ~ds (int ~n-holdout)))
         [{:ds _ :n-holdout _}]
         (let [[holdout-n] (eval-if-code [n-holdout seq? number?])]
           (.splitTestAndTrain ds holdout-n))
         [{:ds (_ :guard seq?)
           :percent-train (:or (_ :guard number?)
                               (_ :guard seq?))}]
         (obj-or-code? as-code? `(.splitTestAndTrain ~ds (double ~percent-train)))
         [{:ds _ :percent-train _}]
         (let [[percent-train-n] (eval-if-code [percent-train seq? number?])]
           (.splitTestAndTrain ds percent-train-n))))

(defn squish-to-range!
  "Squeezes input data to a max and a min"
  [& {:keys [ds min-val max-val as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :min-val (:or (_ :guard number?)
                         (_ :guard seq?))
           :max-val (:or (_ :guard number?)
                         (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.squishToRange (double ~min-val) (double ~max-val))))
         :else
         (let [[min-n max-n] (eval-if-code [min-val seq? number?]
                                           [max-val seq? number?])]
           (doto ds (.squishToRange min-n max-n)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn as-list
  "Extract each example in the DataSet into its own DataSet object,
  and return all of them as a list"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.asList ~ds))
         :else
         (.asList ds)))

(defn has-mask-arrays?
  "does this dataset contain any mask arrays?"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.hasMaskArrays ~ds))
         :else
         (.hasMaskArrays ds)))

(defn new-ds-iter
  "creates an iterator for the supplied dataset"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.iterator ~ds))
         :else
         (.iterator ds)))

(defn load-ds!
  "loads a dataset from a given input stream or a file path

  :in (InputStream), a source to load a dataset from

  :file-path (str), the path to a file containing the dataset"
  [& {:keys [file-path in as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:file-path (:or (_ :guard string?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.load (clojure.java.io/as-file ~file-path)))
         [{:file-path _}]
         (.load (clojure.java.io/as-file file-path))
         [{:in (_ :guard seq?)}]
         (obj-or-code? as-code? `(.load ~in))
         :else
         (.load in)))

(defn sample-ds
  "Sample with/without replacement and a given/random rng"
  [& {:keys [ds n-samples with-replacement? seed as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :n-samples (:or (_ :guard number?)
                           (_ :guard seq?))
           :with-replacement? (:or (_ :guard boolean?)
                                   (_ :guard seq?))
           :seed (:or (_ :guard number?)
                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.sample ~ds (int ~n-samples) (new Random ~seed) ~with-replacement?))
         [{:ds _ :n-samples _ :with-replacement? _ :seed _}]
         (let [[samples-n replacement-b seed-n]
               (eval-if-code [n-samples seq? number?]
                             [with-replacement? seq? boolean?]
                             [seed seq? number?])]
           (.sample ds samples-n (new Random seed-n) replacement-b))
         [{:ds (_ :guard seq?)
           :n-samples (:or (_ :guard number?)
                           (_ :guard seq?))
           :seed (:or (_ :guard number?)
                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.sample ~ds (int ~n-samples) (new Random ~seed)))
         [{:ds _ :n-samples _  :seed _}]
         (let [[samples-n seed-n] (eval-if-code [n-samples seq? number?]
                                                [seed seq? number?])]
           (.sample ds samples-n (new Random seed-n)))
         [{:ds (_ :guard seq?)
           :n-samples (:or (_ :guard number?)
                           (_ :guard seq?))
           :with-replacement? (:or (_ :guard boolean?)
                                   (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.sample ~ds (int ~n-samples) ~with-replacement?))
         [{:ds _ :n-samples _ :with-replacement? _}]
         (let [[samples-n replacement-b] (eval-if-code [n-samples seq? number?]
                                                       [with-replacement? seq? boolean?])]

              (.sample ds samples-n replacement-b))
         [{:ds (_ :guard seq?)
           :n-samples (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.sample ~ds (int ~n-samples)))
         :else
         (let [[samples-n] (eval-if-code [n-samples seq? number?])]
           (.sample ds samples-n))))

(defn save-ds!
  "saves a datset to a given file or output stream

  :out (OutputStream), an output stream to save the dataset to

  :file-path (str), a string to a file you want to save the dataset in"
  [& {:keys [ds file-path out as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:ds (_ :guard seq?)
           :out (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~ds (.save ~out)))
         [{:ds _
           :out _}]
         (let [[ds-obj out-obj] (eval-if-code [ds seq?] [out seq?])]
           (doto ds-obj (.save out-obj)))
         [{:ds (_ :guard seq?)
           :file-path (:or (_ :guard string?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~ds (.save (clojure.java.io/as-file ~file-path))))
         [{:ds _
           :file-path _}]
         (let [[f-path] (eval-if-code [file-path seq? string?])]
           (doto ds (.save (clojure.java.io/as-file f-path))))))

(defn validate-ds!
  "validates a dataset"
  [ds & {:keys [as-code?]
      :or {as-code? true}}]
  (match [ds]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~ds .validate))
         :else
         (doto ds .validate)))
