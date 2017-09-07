(ns dl4clj.datasets.api.input-splits
  (:import [org.datavec.api.split FileSplit BaseInputSplit CollectionInputSplit
            InputStreamInputSplit ListStringSplit NumberedFileInputSplit
            StringSplit TransformSplit TransformSplit$URITransform]
           [org.datavec.api.io.filters BalancedPathFilter RandomPathFilter]
           [org.datavec.api.io.labels PathLabelGenerator ParentPathLabelGenerator
            PatternPathLabelGenerator]
           [org.datavec.api.writable Writable]
           [org.datavec.api.split InputSplit])
  (:require [dl4clj.utils :refer [array-of]]
            [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; path filters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn filter-paths
  "applies the filtering based on the supplied path-filter

   :path-filter (obj), a path filter object

   :paths (collection of strings), uri paths to be filtered

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/filters/PathFilter.html"
  [& {:keys [path-filter paths]
      :as opts}]
  (match [opts]
         [{:path-filter (_ :guard seq?)
           :paths (:or (_ :guard coll?)
                       (_ :guard seq?))}]
         `(.filter ~path-filter
                   (array-of :data (mapv #(java.net.URI/create %) ~paths)
                             :java-type java.net.URI))
         :else
         (let [uris (mapv #(java.net.URI/create %) paths)]
           (.filter path-filter (array-of :data uris
                                          :java-type java.net.URI)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; label generators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-label-for-path
  "used to infer the label of a file directly from the path of a file

  :label-generator (label-generator), call either new-parent-path-label-generator or
   new-pattern-path-label-generator

  :path (string or uri), the file path

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/PathLabelGenerator.html"
  [& {:keys [label-generator path]
      :as opts}]
  (match [opts]
         [{:label-generator (_ :guard seq?)
           :path (:or (_ :guard string?)
                      (_ :guard seq?))}]
         `(.getLabelForPath ~label-generator ~path)
         :else
         (.getLabelForPath label-generator path)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input splits, see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputSplit.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn length
  "Length of the split"
  [split]
  (match [split]
         [(_ :guard seq?)]
         `(.length ~split)
         :else
         (.length split)))

(defn locations
  "Locations of the splits"
  [split]
  (match [split]
         [(_ :guard seq?)]
         `(.locations ~split)
         :else
         (.locations split)))

(defn locations-iterator
  "returns the URI of the iterator assocaited with this filesplit"
  [split]
  (match [split]
         [(_ :guard seq?)]
         `(.locationsIterator ~split)
         :else
         (.locationsIterator split)))

(defn locations-path-iterator
  "returns the file path of an iterator associated with this filesplit"
  [split]
  (match [split]
         [(_ :guard seq?)]
         `(.locationsPathIterator ~split)
         :else
         (.locationsPathIterator split)))

(defn reset-input-split!
  "Reset the InputSplit without reinitializing it from scratch."
  [split]
  (match [split]
         [(_ :guard seq?)]
         `(doto ~split .reset)
         :else
         (doto split .reset)))

(defn read-fields!
  "Deserialize the fields of this object from in."
  [& {:keys [split in]
      :as opts}]
  (match [opts]
         [{:split (_ :guard seq?)
           :in (_ :guard seq?)}]
         `(doto ~split (.readFields ~in))
         :else
         (doto split (.readFields in))))

(defn write!
  "Serialize the fields of this object to out."
  [& {:keys [split out-put-path]
      :as opts}]
  (match [opts]
         [{:split (_ :guard seq?)
           :out-put-path (_ :guard seq?)}]
         `(doto ~split (.write ~out-put-path))
         :else
         (doto split (.write out-put-path))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; base input split, see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/BaseInputSplit.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
  [& {:keys [split path-filter weights]
      :as opts}]
  (match [opts]
         [{:split (_ :guard seq?)
           :path-filter (_ :guard seq?)
           :weights (:or (_ :guard coll?)
                         (_ :guard seq?))}]
         `(.sample ~split ~path-filter (double-array ~weights))
         :else
         (.sample split path-filter (double-array weights))))

(defn to-double!
  "convert writable to double"
  [writeable]
  (match [writeable]
         [(_ :guard seq?)]
         `(.toDouble ~writeable)
         :else
         (.toDouble writeable)))

(defn to-float!
  "convert writeable to float"
  [writeable]
  (match [writeable]
         [(_ :guard seq?)]
         `(.toFloat ~writeable)
         :else
         (.toFloat writeable)))

(defn to-int!
  "convert writeable to int"
  [writeable]
  (match [writeable]
         [(_ :guard seq?)]
         `(.toInt ~writeable)
         :else
         (.toInt writeable)))

(defn to-long!
  "convert writeable to long"
  [writeable]
  (match [writeable]
         [(_ :guard seq?)]
         `(.toLong ~writeable)
         :else
         (.toLong writeable)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file split
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-root-dir
  "returns the root directory

  used with file-split"
  [split]
  (match [split]
         [(_ :guard seq?)]
         `(.getRootDir ~split)
         :else
         (.getRootDir split)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input stream input split
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-is
  "return the input stream

  used with input-stream-input-split"
  [input-stream-input-split]
  (match [input-stream-input-split]
         [(_ :guard seq?)]
         `(.getIs ~input-stream-input-split)
         :else
         (.getIs input-stream-input-split)))

(defn set-is!
  "sets the input stream for an input-stream-input-split

  :is is a java.io.InputStream

  used with input-stream-input-split"
  [& {:keys [input-stream-input-split is]
      :as opts}]
  (match [opts]
         [{:input-stream-input-split (_ :guard seq?)
           :is (_ :guard seq?)}]
         `(doto ~input-stream-input-split (.setIs ~is))
         :else
         (doto input-stream-input-split (.setIs is))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; string splits
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-list-string-split-data
  "returns the string data contained within a string split or a list string split"
  [list-string-split]
  (match [list-string-split]
         [(_ :guard seq?)]
         `(.getData ~list-string-split)
         :else
         (.getData list-string-split)))
