(ns dl4clj.datasets.api.rearrange
  (:import [org.deeplearning4j.datasets.rearrange LocalUnstructuredDataFormatter])
  (:require [clojure.core.match :refer [match]]))


;; untested
(defn get-new-destination
  "sets a new destination for saving a rearranged dataset

  :unstructured-formatter (formatter), the formatter created by new-unstructured-formatter

  :file-path (str), the new file path to save to

  :train? (boolean), is this for training or testing"
  [& {:keys [unstructured-formatter file-path train?]
      :as opts}]
  (match [opts]
         [{:unstructured-formatter (_ :guard seq?)
           :file-path (:or (_ :guard string?)
                           (_ :guard seq?))
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))}]
         `(.getNewDestination ~unstructured-formatter ~file-path ~train?)
         :else
         (.getNewDestination unstructured-formatter file-path train?)))

(defn get-name-label
  "returns the name of the label assigned to a file

  :unstructured-formatter (formatter), the formatter created by new-unstructured-formatter

  :file-path (str), one of the files used to create the unstructured-formatter"
  [& {:keys [unstructured-formatter file-path]
      :as opts}]
  (match [opts]
         [{:unstructured-formatter (_ :guard seq?)
           :file-path (:or (_ :guard string?)
                           (_ :guard seq?))}]
         `(.getNameLabel ~unstructured-formatter ~file-path)
         :else
         (.getNameLabel unstructured-formatter file-path)))

(defn get-num-examples-total
  "returns the total number of examples in the dataset"
  [unstructured-formatter]
  (match [unstructured-formatter]
         [(_ :guard seq?)]
         `(.getNumExamplesTotal ~unstructured-formatter)
         :else
         (.getNumExamplesTotal unstructured-formatter)))

(defn get-num-examples-to-train-on
  "returns the number of examples in the training split of the dataset"
  [unstructured-formatter]
  (match [unstructured-formatter]
         [(_ :guard seq?)]
         `(.getNumExamplesToTrainOn ~unstructured-formatter)
         :else
         (.getNumExamplesToTrainOn unstructured-formatter)))

(defn get-num-test-examples
  "returns the number of examples in the testing split of the dataset"
  [unstructured-formatter]
  (match [unstructured-formatter]
         [(_ :guard seq?)]
         `(.getNumTestExamples ~unstructured-formatter)
         :else
         (.getNumTestExamples unstructured-formatter)))

(defn get-path-label
  "returns the label assigned to a file path

  :unstructured-formatter (formatter), the formatter created by new-unstructured-formatter

  :file-path (str), a file path used in the creation of the dataset"
  [& {:keys [unstructured-formatter file-path]
      :as opts}]
  (match [opts]
         [{:unstructured-formatter (_ :guard seq?)
           :file-path (:or (_ :guard string?)
                           (_ :guard seq?))}]
         `(.getPathLabel ~unstructured-formatter ~file-path)
         :else
         (.getPathLabel unstructured-formatter file-path)))

(defn get-test
  "returns the file containing the test split of the dataset"
  [unstructured-formatter]
  (match [unstructured-formatter]
         [(_ :guard seq?)]
         `(.getTest ~unstructured-formatter)
         :else
         (.getTest unstructured-formatter)))

(defn get-train
  "returns the file containing the training split of the dataset"
  [unstructured-formatter]
  (match [unstructured-formatter]
         [(_ :guard seq?)]
         `(.getTrain ~unstructured-formatter)
         :else
         (.getTrain unstructured-formatter)))

(defn rearrange!
  "rearranges the dataset and returns the formatter"
  [unstructured-formatter]
  (match [unstructured-formatter]
         [(_ :guard seq?)]
         `(doto ~unstructured-formatter .rearrange)
         :else
         (doto unstructured-formatter .rearrange)))
