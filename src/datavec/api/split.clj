(ns datavec.api.split
  (:import [org.datavec.api.split FileSplit BaseInputSplit CollectionInputSplit
            InputStreamInputSplit])
  (:require [clojure.java.io :as io]
            [dl4clj.nn.conf.utils :refer [contains-many?]]))

;; TODO
;; add doc-strings
;; implement these classes
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/BaseInputSplit.html
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/CollectionInputSplit.html
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputStreamInputSplit.html
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/ListStringSplit.html
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/NumberedFileInputSplit.html
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/StringSplit.html
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/TransformSplit.html

;; basically classes that implement this interface:
;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputSplit.html

(defn split-type
  "dispatch fn for input-split"
  [opts]
  (first (keys opts)))

(defmulti input-split
  "Multimethod that builds an input split based on the supplied type and opts"
  split-type)

(defmethod input-split :file-split [opts]
  (let [config (:file-split opts)
        {root :root-dir
         rng :rng-seed
         fmt :allow-format
         recursive? :recursive?} config]
    (cond (contains-many? config :root-dir :allow-format :rng-seed)
          (FileSplit. (io/as-file root) fmt rng)
          (contains-many? config :root-dir :allow-format :recursive?)
          (FileSplit. (io/as-file root) fmt recursive?)
          (contains-many? config :root-dir :allow-format)
          (FileSplit. (io/as-file root) fmt)
          (contains-many? config :root-dir :rng-seed)
          (FileSplit. (io/as-file root) rng)
          :else
          (FileSplit. (io/as-file root)))))

(defmethod input-split :collection-input-split [opts]
  (let [config (:collection-input-split opts)
        coll-of-uris (:collection config)]
    (CollectionInputSplit. coll-of-uris)))

(defmethod input-split :input-stream-input-split [opts]
  ;; file-path can be a file, a string, or a uri
  ;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputStreamInputSplit.html
  ;; "Input stream input split. The normal pattern is reading the whole input stream and turning that in to a record. This is meant for streaming raw data rather than normal mini batch pre processing."
  ;; Instantiate with the given file as a uri
  (let [config (:input-stream-input-split opts)
        {is :in-stream
         path :file-path} config]
    (if (contains? config :file-path)
      (InputStreamInputSplit. is path)
      (InputStreamInputSplit. is))))

;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/ListStringSplit.html

;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/NumberedFileInputSplit.html

;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/StringSplit.html

;; https://deeplearning4j.org/datavecdoc/org/datavec/api/split/TransformSplit.html

(defn new-filesplit
  "File input split. Splits up a root directory in to files.

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/FileSplit.html"
  [{:keys [root-dir rng-seed allow-format recursive?]
      :as opts}]
  (input-split {:file-split opts}))

(defn get-root-dir
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

(defn initialize
  [split]
  (.initialize split))

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

(defn read-fields!
  "Deserialize the fields of this object from in."
  [split in]
  (doto split (.readFields in)))

(defn reset!
  "Reset the InputSplit without reinitializing it from scratch."
  [split]
  (doto split (.reset)))

(defn write!
  "Serialize the fields of this object to out."
  [split out-put-path]
  (doto split (.write out-put-path)))

(defn sample
  "Samples the locations based on the path-filter and splits
  the result into an array of input-pplit objects,
  with sizes proportional to the weights.

  args are:
  :path-filter (map) to modify the locations in some way (null == as is)
  :weights (double array) to split the locations into multiple InputSplit"
  ;; write multimethods for the path-filter config map
  ;; https://deeplearning4j.org/datavecdoc/org/datavec/api/io/filters/PathFilter.html
  [& {:keys [split path-filter weights-array]}]
  (.sample split path-filter weights-array))

(defn to-double!
  "convert writable to double"
  [writeable]
  (.toDouble writeable))

(defn to-float!
  "convert writeable to float"
  [writeable]
  (.toFloat writeable))

(defn to-int!
  "convert writeable to int"
  [writeable]
  (.toInt writeable))

(defn to-long!
  "convert writeable to long"
  [writeable]
  (.toLong writeable))
