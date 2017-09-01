(ns dl4clj.datasets.api.rearrange
  (:import [org.deeplearning4j.datasets.rearrange LocalUnstructuredDataFormatter]))

;; untested
(defn get-new-destination
  "sets a new destination for saving a rearranged dataset

  :unstructured-formatter (formatter), the formatter created by new-unstructured-formatter

  :file-path (str), the new file path to save to

  :train? (boolean), is this for training or testing"
  [& {:keys [unstructured-formatter file-path train?]}]
  (.getNewDestination unstructured-formatter file-path train?))

(defn get-name-label
  "returns the name of the label assigned to a file

  :unstructured-formatter (formatter), the formatter created by new-unstructured-formatter

  :file-path (str), one of the files used to create the unstructured-formatter"
  [& {:keys [unstructured-formatter file-path]}]
  (.getNameLabel unstructured-formatter file-path))

(defn get-num-examples-total
  "returns the total number of examples in the dataset"
  [unstructured-formatter]
  (.getNumExamplesTotal unstructured-formatter))

(defn get-num-examples-to-train-on
  "returns the number of examples in the training split of the dataset"
  [unstructured-formatter]
  (.getNumExamplesToTrainOn unstructured-formatter))

(defn get-num-test-examples
  "returns the number of examples in the testing split of the dataset"
  [unstructured-formatter]
  (.getNumTestExamples unstructured-formatter))

(defn get-path-label
  "returns the label assigned to a file path

  :unstructured-formatter (formatter), the formatter created by new-unstructured-formatter

  :file-path (str), a file path used in the creation of the dataset"
  [& {:keys [unstructured-formatter file-path]}]
  (.getPathLabel unstructured-formatter file-path))

(defn get-test
  "returns the file containing the test split of the dataset"
  [unstructured-formatter]
  (.getTest unstructured-formatter))

(defn get-train
  "returns the file containing the training split of the dataset"
  [unstructured-formatter]
  (.getTrain unstructured-formatter))

(defn rearrange!
  "rearranges the dataset and returns the formatter"
  [unstructured-formatter]
  (doto unstructured-formatter (.rearrange)))
