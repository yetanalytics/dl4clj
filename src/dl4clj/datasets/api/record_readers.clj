(ns dl4clj.datasets.api.record-readers
  (:import [org.datavec.api.records.reader RecordReader SequenceRecordReader]
           [org.datavec.api.conf Configurable]
           [java.io Closeable])
  (:require [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-conf-rr
  "Return the configuration used by this record reader"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.getConf ~rr)
         :else
         (.getConf rr)))

(defn get-labels-reader
  "List of label strings"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.getLabels ~rr)
         :else
         (.getLabels rr)))

(defn next-record!
  "Get the next record"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.next ~rr)
         :else
         (.next rr)))

(defn next-record-with-meta!
  "Similar to next!, but returns a record reader,
  that may include metadata such as the source of the data"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.nextRecord ~rr)
         :else
         (.nextRecord rr)))

(defn get-listeners-rr
  "Get the record listeners for this record reader."
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.getListeners ~rr)
         :else
         (.getListeners rr)))

(defn next-seq!
  "Similar to sequence-record, but returns a Record object,
  that may include metadata such as the source of the data"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.nextSequence ~rr)
         :else
         (.nextSequence rr)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn set-listeners-rr!
  "Set the record listeners for this record reader."
  [& {:keys [rr listeners]
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :listeners (:or (_ :guard coll?)
                           (_ :guard seq?))}]
         `(doto ~rr (.setListeners ~listeners))
         :else
         (doto rr (.setListeners listeners))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn has-next-record?
  "Check whether there are anymore records"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(.hasNext ~rr)
         :else
         (.hasNext rr)))

(defn reset-rr!
  "Reset record reader iterator"
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(doto ~rr .reset)
         :else
         (doto rr .reset)))

(defn load-record
  "Load the record from the given data-in-stream
  Unlike next!, the internal state of the record-reader is not modified
  Implementations of this method should not close the DataInputStream"
  [& {:keys [rr uri data-in-stream]
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :uri (:or (_ :guard string?)
                     (_ :guard seq?))
           :data-in-stream (_ :guard seq?)}]
         `(.record ~rr (java.net.URI. ~uri) ~data-in-stream)
         :else
         (.record rr uri data-in-stream)))

(defn initialize-rr!
  "will need to be updated when other rr's are implemented

  :input-split (input split) the split that defines the range of records to read
   -see datavec.api.split

  :conf (map) a configuration for initialization

  this is how data actually gets into the record reader"
  [& {:keys [rr input-split conf]
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :input-split (_ :guard seq?)
           :conf (_ :guard seq?)}]
         `(doto ~rr (.initialize ~conf ~input-split))
         [{:rr _
           :input-split _
           :conf _}]
         (doto rr (.initialize conf input-split))
         [{:rr (_ :guard seq?)
           :input-split (_ :guard seq?)}]
         `(doto ~rr (.initialize ~input-split))
         [{:rr _
           :input-split _}]
         (doto rr (.initialize input-split))))

(defn load-from-meta-data-rr
  "loads a single or multiple record(s) from a given RecordMetaData instance (or list of)
   -see TBD for record meta data (not yet implemented)
  https://deeplearning4j.org/datavecdoc/org/datavec/api/records/metadata/RecordMetaData.html"
  [& {:keys [rr record-meta-data]
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :record-meta-data (_ :guard seq?)}]
         `(.loadFromMetaData ~rr ~record-meta-data)
         :else
         (.loadFromMetaData rr record-meta-data)))

(defn load-seq-from-meta-data-rr
  "Load multiple (or single) sequence record(s) from the given list of record meta data instances

  number of sequences is determined by how record meta data is passed
   - as a list of instances, multiple
   - a single instance, single seq"
  [& {:keys [rr record-meta-data]
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :record-meta-data (_ :guard seq?)}]
         `(.loadSequenceFromMetaData ~rr ~record-meta-data)
         :else
         (.loadSequenceFromMetaData rr record-meta-data)))

(defn close!
  "Closes this stream and releases any system resources associated with it."
  [rr]
  (match [rr]
         [(_ :guard seq?)]
         `(doto ~rr .close)
         :else
         (doto rr .close)))

(defn sequence-record
  "Load a sequence record from the given data-in-stream
  Unlike next-data-record the internal state of the record-reader is not modified
  Implementations of this method should not close the data-in-stream"
  [& {:keys [rr uri data-in-stream]
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :uri (:or (_ :guard string?)
                     (_ :guard seq?))
           :data-in-stream (_ :guard seq?)}]
         `(.sequenceRecord ~rr (java.net.URI. ~uri) ~data-in-stream)
         [{:rr _
           :uri _
           :data-in-stream _}]
         (.sequenceRecord rr uri data-in-stream)
         [{:rr (_ :guard seq?)}]
         `(.sequenceRecord ~rr)
         [{:rr _}]
         (.sequenceRecord rr)))
