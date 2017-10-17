(ns dl4clj.datasets.api.record-readers
  (:import [org.datavec.api.records.reader RecordReader SequenceRecordReader]
           [org.datavec.api.conf Configurable]
           [java.io Closeable])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-conf-rr
  "Return the configuration used by this record reader"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getConf ~rr))
         :else
         (.getConf rr)))

(defn get-labels-reader
  "List of label strings"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLabels ~rr))
         :else
         (.getLabels rr)))

(defn next-record!
  "Get the next record"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.next ~rr))
         :else
         (.next rr)))

(defn next-record-with-meta!
  "Similar to next!, but returns a record reader,
  that may include metadata such as the source of the data"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.nextRecord ~rr))
         :else
         (.nextRecord rr)))

(defn get-listeners-rr
  "Get the record listeners for this record reader."
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getListeners ~rr))
         :else
         (.getListeners rr)))

(defn next-seq!
  "Similar to sequence-record, but returns a Record object,
  that may include metadata such as the source of the data"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.nextSequence ~rr))
         :else
         (.nextSequence rr)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn set-listeners-rr!
  "Set the record listeners for this record reader."
  [& {:keys [rr listeners as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :listeners (:or (_ :guard coll?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~rr (.setListeners ~listeners)))
         [{:rr (_ :guard seq?)
           :listeners _}]
         (doto (eval rr) (.setListeners (if (coll? listeners)
                                          listeners
                                          [listeners])))
         [{:rr _
           :listeners (_ :guard coll?)}]
         (doto rr (.setListeners (map eval listeners)))
         [{:rr _
           :listeners (_ :guard seq?)}]
         (doto rr (.setListeners [(eval listeners)]))
         :else
         (doto rr (.setListeners listeners))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn has-next-record?
  "Check whether there are anymore records"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.hasNext ~rr))
         :else
         (.hasNext rr)))

(defn reset-rr!
  "Reset record reader iterator"
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~rr .reset))
         :else
         (doto rr .reset)))

(defn load-record
  "Load the record from the given data-in-stream
  Unlike next!, the internal state of the record-reader is not modified
  Implementations of this method should not close the DataInputStream"
  [& {:keys [rr uri data-in-stream as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :uri (:or (_ :guard string?)
                     (_ :guard seq?))
           :data-in-stream (_ :guard seq?)}]
         ;; this will break if the uri seq evals to a uri
         ;; need a way to account for the return type
         (obj-or-code? as-code? `(.record ~rr (java.net.URI. ~uri) ~data-in-stream))
         [{:rr _
           :uri (_ :guard string?)
           :data-in-stream (_ :guard seq?)}]
         (.record rr (java.net.URI. uri) (eval data-in-stream))
         [{:rr (_ :guard seq?)
           :uri (_ :guard string?)
           :data-in-stream _}]
         (.record (eval rr) (java.net.URI. uri) data-in-stream)
         ;; there might be another condition im missing here
         :else
         (.record rr (java.net.URI. uri) data-in-stream)))

(defn initialize-rr!
  "will need to be updated when other rr's are implemented

  :input-split (input split) the split that defines the range of records to read
   -see datavec.api.split

  :conf (map) a configuration for initialization
   - need to determine what this configuration is and if it should be included at all

  this is how data actually gets into the record reader"
  [& {:keys [rr input-split conf as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :input-split (_ :guard seq?)
           :conf (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~rr (.initialize ~conf ~input-split)))
         ;; this level of guarding may be unncessary
         ;; should I be this concerned about the user shooting themself in the foot?
         [{:rr (_ :guard seq?)
           :input-split _
           :conf (_ :guard seq?)}]
         ;; input-split is an object
         (doto (eval rr) (.initialize (eval conf) input-split))
         [{:rr (_ :guard seq?)
           :input-split (_ :guard seq?)
           :conf _}]
         ;; conf is an object
         (doto (eval rr) (.initialize conf (eval input-split)))
         [{:rr _
           :input-split (_ :guard seq?)
           :conf (_ :guard seq?)}]
         ;; record reader is an object
         (doto rr (.initialize (eval conf) (eval input-split)))
         [{:rr (_ :guard seq?)
           :input-split _
           :conf _}]
         ;; only rr isnt an obj
         (doto (eval rr) (.initialize conf input-split))
         [{:rr _
           :input-split (_ :guard seq?)
           :conf _}]
         ;; only is isn't an obj
         (doto rr (.initialize conf (eval input-split)))
         [{:rr _
           :input-split _
           :conf (_ :guard seq?)}]
         ;; only conf isnt an obj
         (doto rr (.initialize (eval conf) input-split))
         [{:rr _
           :input-split _
           :conf _}]
         ;; all are objs
         (doto rr (.initialize conf input-split))
         [{:rr (_ :guard seq?)
           :input-split (_ :guard seq?)}]
         ;; neither are objs
         (obj-or-code? as-code? `(doto ~rr (.initialize ~input-split)))
         [{:rr (_ :guard seq?)
           :input-split _}]
         ;; is is an obj but rr is not
         (doto (eval rr) (.initialize input-split))
         [{:rr _
           :input-split (_ :guard seq?)}]
         ;; rr is an obj but is is not
         (doto rr (.initialize (eval input-split)))
         ;; both are objs
         [{:rr _
           :input-split _}]
         (doto rr (.initialize input-split))))

(defn load-from-meta-data-rr
  "loads a single or multiple record(s) from a given RecordMetaData instance (or list of)
   -see TBD for record meta data (not yet implemented)
  https://deeplearning4j.org/datavecdoc/org/datavec/api/records/metadata/RecordMetaData.html"
  [& {:keys [rr record-meta-data as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :record-meta-data (_ :guard seq?)}]
         (obj-or-code? as-code? `(.loadFromMetaData ~rr ~record-meta-data))
         :else
         (.loadFromMetaData rr record-meta-data)))

(defn load-seq-from-meta-data-rr
  "Load multiple (or single) sequence record(s) from the given list of record meta data instances

  number of sequences is determined by how record meta data is passed
   - as a list of instances, multiple
   - a single instance, single seq"
  [& {:keys [rr record-meta-data as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :record-meta-data (_ :guard seq?)}]
         (obj-or-code? as-code? `(.loadSequenceFromMetaData ~rr ~record-meta-data))
         :else
         (.loadSequenceFromMetaData rr record-meta-data)))

(defn close!
  "Closes this stream and releases any system resources associated with it."
  [rr & {:keys [as-code?]
         :or {as-code? true}}]
  (match [rr]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~rr .close))
         :else
         (doto rr .close)))

(defn sequence-record
  "Load a sequence record from the given data-in-stream
  Unlike next-data-record the internal state of the record-reader is not modified
  Implementations of this method should not close the data-in-stream"
  ;; need to note that this fn assumes uri is a string
  [& {:keys [rr uri data-in-stream as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rr (_ :guard seq?)
           :uri (:or (_ :guard string?)
                     (_ :guard seq?))
           :data-in-stream (_ :guard seq?)}]
         (obj-or-code? as-code? `(.sequenceRecord ~rr (java.net.URI. ~uri) ~data-in-stream))
         [{:rr (_ :guard seq?)
           :uri (:or (_ :guard string?)
                     (_ :guard seq?))
           :data-in-stream _}]
         (.sequenceRecord (eval rr) (java.net.URI. uri) data-in-stream)
         [{:rr _
           :uri (:or (_ :guard string?)
                     (_ :guard seq?))
           :data-in-stream (_ :guard seq?)}]
         (.sequenceRecord rr (java.net.URI. uri) (eval data-in-stream))
         [{:rr _
           :uri _
           :data-in-stream _}]
         (.sequenceRecord rr (java.net.URI. uri) data-in-stream)
         [{:rr (_ :guard seq?)}]
         (obj-or-code? as-code? `(.sequenceRecord ~rr))
         :else
         (.sequenceRecord rr)))
