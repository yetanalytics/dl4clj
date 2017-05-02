(ns datavec.api.split
  (:import [org.datavec.api.split FileSplit])
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

(defn new-filesplit
  "File input split. Splits up a root directory in to files."
  [{:keys [root-dir rng-seed allow-format
             recursive?]
      :as opts}]
  (cond (contains-many? opts :root-dir :allow-format :rng-seed)
        (FileSplit. (io/as-file root-dir) allow-format rng-seed)
        (contains-many? opts :root-dir :allow-format :recursive?)
        (FileSplit. (io/as-file root-dir) allow-format recursive?)
        (contains-many? opts :root-dir :allow-format)
        (FileSplit. (io/as-file root-dir) allow-format)
        (contains-many? opts :root-dir :rng-seed)
        (FileSplit. (io/as-file root-dir) rng-seed)
        :else
        (FileSplit. (io/as-file root-dir))))

(defn get-root-dir
  [file-split]
  (.getRootDir file-split))

(defn initialize
  [file-split]
  (.initialize file-split))

(defn length
  "Length of the split"
  [file-split]
  (.length file-split))

(defn read-fields
  "Deserialize the fields of this object from in."
  [file-split in]
  (.readFields file-split in))

(defn reset
  "Reset the InputSplit without reinitializing it from scratch."
  [file-split]
  (.reset file-split))

(defn write
  "Serialize the fields of this object to out."
  [file-split out-put-path]
  (.write file-split out-put-path))
