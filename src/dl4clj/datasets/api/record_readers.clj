(ns dl4clj.datasets.api.record-readers
  (:import [org.datavec.api.records.reader RecordReader SequenceRecordReader]
           [org.datavec.api.conf Configurable]
           [java.io Closeable])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn get-conf-rr
  "Return the configuration used by this record reader"
  [rr]
  (.getConf rr))

(defn set-conf-rr!
  "Set the configuration to be used by this record reader."
  [& {:keys [rr conf]}]
  (doto rr
    (.setConf conf)))

(defn get-labels-reader
  "List of label strings"
  [rr]
  (.getLabels rr))

(defn next-record!
  "Get the next record"
  [rr]
  (.next rr))

(defn next-record-with-meta!
  "Similar to next!, but returns a record reader,
  that may include metadata such as the source of the data"
  [rr]
  (.nextRecord rr))

(defn has-next-record?
  "Check whether there are anymore records"
  [rr]
  (.hasNext rr))

(defn reset-rr!
  "Reset record reader iterator"
  [rr]
  (doto rr
    (.reset)))

(defn get-listeners-rr
  "Get the record listeners for this record reader."
  [rr]
  (.getListeners rr))

(defn set-listeners-rr!
  "Set the record listeners for this record reader."
  [& {:keys [rr listeners]}]
  (doto rr
    (.setListeners listeners)))

(defn load-record
  "Load the record from the given data-in-stream
  Unlike next!, the internal state of the record-reader is not modified
  Implementations of this method should not close the DataInputStream"
  [& {:keys [rr uri data-in-stream]}]
  (.record rr uri data-in-stream ))

(defn initialize-rr!
  ;; refactor once datavec.api.split is refactored (:input-split)
  "will need to be updated when other rr's are implemented

  :input-split (input split) the split that defines the range of records to read
   -see datavec.api.split
  :conf (map) a configuration for initialization

  this is how data actually gets into the record reader"
  [& {:keys [rr input-split conf]
      :as opts}]
  (if conf
    (doto rr (.initialize conf input-split))
    (doto rr (.initialize input-split))))

(defn load-from-meta-data-rr
  "loads a single or multiple record(s) from a given RecordMetaData instance (or list of)
   -see TBD for record meta data (not yet implemented)
  https://deeplearning4j.org/datavecdoc/org/datavec/api/records/metadata/RecordMetaData.html"
  [& {:keys [rr record-meta-data]}]
  (.loadFromMetaData rr record-meta-data))

(defn load-seq-from-meta-data-rr
  "Load multiple (or single) sequence record(s) from the given list of record meta data instances

  number of sequences is determined by how record meta data is passed
   - as a list of instances, multiple
   - a single instance, single seq"
  [& {:keys [rr record-meta-data]}]
  (.loadSequenceFromMetaData rr record-meta-data))

(defn close!
  "Closes this stream and releases any system resources associated with it."
  [rr]
  (doto rr
    (.close)))

(defn next-seq!
  "Similar to sequence-record, but returns a Record object,
  that may include metadata such as the source of the data"
  [rr]
  (.nextSequence rr))

(defn sequence-record
  "Load a sequence record from the given data-in-stream
  Unlike next-data-record the internal state of the record-reader is not modified
  Implementations of this method should not close the data-in-stream"
  [& {:keys [rr uri data-in-stream]
      :as opts}]
  (if (contains-many? opts :uri :data-in-stream)
    (.sequenceRecord rr uri data-in-stream)
    (.sequenceRecord rr)))
