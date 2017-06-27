(ns dl4clj.datasets.api.input-splits
  (:import [org.datavec.api.split FileSplit BaseInputSplit CollectionInputSplit
            InputStreamInputSplit ListStringSplit NumberedFileInputSplit
            StringSplit TransformSplit TransformSplit$URITransform]
           [org.datavec.api.io.filters BalancedPathFilter RandomPathFilter]
           [org.datavec.api.io.labels PathLabelGenerator ParentPathLabelGenerator
            PatternPathLabelGenerator]
           [org.datavec.api.split InputSplit])
  (:require [dl4clj.utils :refer [array-of]]))

(defn filter-paths
  ":path-filter (obj), a path filter object
   :paths (collection of URIs), uri paths to be filtered"
  [& {:keys [path-filter paths]}]
  (.filter path-filter (array-of :data paths
                                 :java-type java.net.URI)))

(defn get-label-for-path
  "used to infer the label of a file directly from the path of a file

  :label-generator (label-generator), call either new-parent-path-label-generator or
   new-pattern-path-label-generator

  :path (string or uri), the file path

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/PathLabelGenerator.html"
  [& {:keys [label-generator path]}]
  (.getLabelForPath label-generator path))

(defn get-root-dir
  "returns the root directory"
  [split]
  (.getRootDir split))

(defn get-is
  "return the input stream"
  [input-stream-input-split]
  (.getIs input-stream-input-split))

(defn set-is!
  "sets the input stream for an input-stream-input-split

  :is is a java.io.InputStream"
  [& {:keys [input-stream-input-split is]}]
  (doto input-stream-input-split (.setIs is)))

(defn get-list-string-split-data
  "returns the string data contained within a string split or a list string split"
  [list-string-split]
  (.getData list-string-split))

(defn sample
  "Samples the locations based on the path-filter and splits
  the result into an array of input-pplit objects,
  with sizes proportional to the weights.
   - you can interact with the returned array like you would a vector
     - ie. first, second, count ....

  args are:
  :path-filter (path-filter), call either new-balanced-path-filter or new-random-path-filter
   -to modify the locations (file paths) in some way (null == as is)

  :weights (coll) to split the locations into multiple InputSplit"
  [& {:keys [split path-filter weights]}]
  (.sample split path-filter (double-array weights)))

(defn length
  "Length of the split"
  [split]
  (.length split))

(defn locations
  "Locations of the splits"
  [split]
  (.locations split))

(defn locations-iterator
  "returns the URI of the iterator assocaited with this filesplit"
  [split]
  (.locationsIterator split))

(defn locations-path-iterator
  "returns the file path of an iterator associated with this filesplit"
  [split]
  (.locationsPathIterator split))

(defn reset-input-split!
  "Reset the InputSplit without reinitializing it from scratch."
  [split]
  (doto split (.reset)))
