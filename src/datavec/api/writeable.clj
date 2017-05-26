(ns ^{:doc "implementation of the writeable interface and its subinterface inputsplit

see: https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html"}
    datavec.api.writeable
  (:refer-clojure :exclude [reset!])
  (:import [org.datavec.api.writable Writable]
           [org.datavec.api.split InputSplit]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; from the Writeable Interface
;; see: https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html
;; these fns are not tested
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn read-fields!
  "Deserialize the fields of this object from in."
  [& {:keys [split in]}]
  (doto split (.readFields in)))

(defn write!
  "Serialize the fields of this object to out."
  [& {:keys [split out-put-path]}]
  (doto split (.write out-put-path)))

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; from the InputSplit Interface
;; see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputSplit.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
