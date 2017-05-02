(ns ^{:doc "see https://deeplearning4j.org/etl-userguide"}
    datavec.api.records.readers
  (:import [org.datavec.api.records.reader BaseRecordReader RecordReader SequenceRecordReader]
           [org.datavec.image.recordreader BaseImageRecordReader ImageRecordReader]
           [org.datavec.api.records.reader.impl.collection CollectionRecordReader]
           [org.datavec.api.records.reader.impl FileRecordReader]
           [org.datavec.api.records.reader.factory RecordReaderFactory
            RecordWriterFactory]
           [org.datavec.api.records.reader.impl.collection CollectionSequenceRecordReader
            ListStringRecordReader]
           [org.datavec.api.records.reader.impl ComposableRecordReader LineRecordReader]
           [org.datavec.api.records.reader.impl.jackson JacksonRecordReader]
           [org.datavec.image.recordreader VideoRecordReader]
           [org.datavec.api.records.reader.impl.csv CSVRecordReader
            CSVNLinesSequenceRecordReader CSVRegexRecordReader]
           [org.datavec.api.records.reader.impl.misc LibSvmRecordReader]
           [org.datavec.api.records.reader.impl.regex RegexLineRecordReader]
           [org.datavec.api.records.reader.impl.misc SVMLightRecordReader]
           [org.datavec.api.records.reader.impl.csv CSVSequenceRecordReader]
           [org.datavec.api.records.reader.impl.misc MatlabRecordReader]
           [org.datavec.api.records.reader.impl.regex RegexSequenceRecordReader])
  (:require [dl4clj.utils :refer [contains-many?]]
            [datavec.api.split :refer [new-filesplit]]))

;; TODO
;; implement other readers

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn record-type
  "dispatch fn for record-reader"
  [opts]
  (first (keys opts)))

(defmulti record-reader
  "Multimethod that builds a record reader based on the supplied type and opts"
  record-type)

(defmethod record-reader :csv-nlines-seq-rr [opts]
  (let [config (:csvn-lines-seq-rr opts)
        {skip-lines :skip-num-lines
         delim :delimiter
         n-lines-per-seq :n-lines-per-seq} config]
    (cond (contains-many? config :skip-num-lines :delimiter :n-lines-per-seq)
          (CSVNLinesSequenceRecordReader. n-lines-per-seq skip-lines delim)
          (contains? config :n-lines-per-seq)
          (CSVNLinesSequenceRecordReader. n-lines-per-seq)
          :else
          (CSVNLinesSequenceRecordReader.))))

(defmethod record-reader :csv-rr [opts]
  (let [config (:csv-rr opts)
        {skip-lines :skip-num-lines
         delim :delimiter
         strip-quotes :strip-quotes} config]
    (cond (contains-many? config :skip-num-lines :delimiter :strip-quotes)
          (CSVRecordReader. skip-lines delim strip-quotes)
          (contains-many? config :skip-num-lines :delimiter)
          (CSVRecordReader. skip-lines delim)
          (contains? config :skip-num-lines)
          (CSVRecordReader. skip-lines)
          :else
          (CSVRecordReader.))))

(defmethod record-reader :csv-seq-rr [opts]
  (let [config (:csv-seq-rr opts)
        {skip-lines :skip-num-lines
         delim :delimiter} config]
    (cond (contains-many? config :skip-num-lines :delimiter)
          (CSVSequenceRecordReader. skip-lines delim)
          (contains? config :skip-num-lines)
          (CSVSequenceRecordReader. skip-lines)
          :else
          (CSVSequenceRecordReader.))))


(defmethod record-reader :file-rr [opts]
  (FileRecordReader.))

(defmethod record-reader :line-rr [opts]
  (LineRecordReader.))

(defmethod record-reader :list-string-rr [opts]
  (ListStringRecordReader.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-csv-nlines-seq-record-reader
  "A CSV Sequence record reader where:
  (a) all time series are in a single file
  (b) each time series is of the same length (specified in constructor)
  (c) no delimiter is used between time series
  For example, with :n-lines-per-seq = 10, lines 0 to 9 are the first time series, 10 to 19 are the second, and so on.

  args are:
  :skip-num-lines (int) number of lines to skip
  :delimiter (str) the delimiter seperating values
  :n-lines-per-seq (int) the number of lines which compose a single series

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVNLinesSequenceRecordReader.html"
  [& {:keys [skip-num-lines delimiter n-lines-per-seq]
      :as opts}]
  (record-reader {:csv-nlines-seq-rr opts}))

(defn new-csv-record-reader
  "Simple csv record reader

  args are:
  :skip-num-lines (int) number of lines to skip
  :delimiter (str) the delimiter seperating values
  :strip-quotes (str) the quote to strip

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRecordReader.html"
  [& {:keys [skip-num-lines delimiter strip-quotes]
      :as opts}]
  (record-reader {:csv-rr opts}))

(defn new-csv-seq-record-reader
  "This reader is intended to read sequences of data in CSV format,
   where each sequence is defined in its own file (and there are multiple files)
   Each line in the file represents one time step

   args are:
   :skip-num-lines (int) number of lines to skip
   :delimiter (str), the delimiter seperating values

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVSequenceRecordReader.html"
  [& {:keys [skip-num-lines delimiter]
      :as opts}]
  (record-reader {:csv-seq-rr opts}))

(defn new-file-record-reader
  "File reader/writer, no args required

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/FileRecordReader.html"
  []
  (record-reader {:file-rr {}}))

(defn new-line-record-reader
  "Reads files line by line, no args required

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/LineRecordReader.html"
  []
  (record-reader {:line-rr {}}))

(defn new-list-string-record-reader
  "Iterates through a list of strings return a record.
  Only accepts a list-string-input-split as input during initialization

  no args needed to call the constructor

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/collection/ListStringRecordReader.html"
  []
  (record-reader {:list-string-rr {}}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record reader shared interaction fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn create-record-writer
  "Factory for creating RecordWriter instance"
  [& {:keys [rw where-to-save]}]
  (.create rw where-to-save))

(defn create-record-reader
  "Factory for creating RecordReader instance"
  [& {:keys [rr where-to-save]}]
  (.create rr where-to-save))

(defn close
  "Closes this stream and releases any system resources associated with it."
  [rr]
  (doto rr
    (.close)))

(defn get-conf
  "Return the configuration used by this record reader"
  [rr]
  (.getConf rr))

(defn get-labels
  "List of label strings"
  [rr]
  (.getLabels rr))

(defn has-next?
  "Check whether there are anymore records"
  [rr]
  (.hasNext rr))

(defn initialize
  ;; refactor once datavec.api.split is refactored (:input-split)
  "will need to be updated when other rr's are implemented

  :input-split (map) the split that defines the range of records to read
  :conf (map) a configuration for initialization

  this is how data actually gets into the record reader"
  [& {:keys [rr input-split conf]
      :as opts}]
  (assert (contains-many? opts :rr :input-split))
  (if (contains? opts conf)
    (.initialize rr conf input-split)
    (.initialize rr input-split)))

(defn load-from-meta-data
  "loads a single or multiple record(s) from a given RecordMetaData instance (or list of)
   -see TBD for record meta data
  https://deeplearning4j.org/datavecdoc/org/datavec/api/records/metadata/RecordMetaData.html"
  [& {:keys [rr record-meta-data]}]
  (.loadFromMetaData rr record-meta-data))

(defn load-seq-from-meta-data
  "Load multiple (or single) sequence record(s) from the given list of record meta data instances

  number of sequences is determined by how record meta data is passed
   - as a list of instances, multiple
   - a single instance, single seq"
  [& {:keys [rr record-meta-data]}]
  (.loadSequenceFromMetaData rr record-meta-data))

(defn next!
  "Get the next record"
  [rr]
  (.next rr))

(defn next-record
  "Similar to next!, but returns a record reader,
  that may include metadata such as the source of the data"
  [rr]
  (.nextRecord rr))

(defn record
  "Load the record from the given data-in-stream
  Unlike next!, the internal state of the record-reader is not modified
  Implementations of this method should not close the DataInputStream"
  [& {:keys [rr uri data-in-stream]}]
  (.record rr uri data-in-stream ))

(defn reset
  "Reset record reader iterator"
  [rr]
  (doto rr
    (.reset)))

(defn set-conf
  "Set the configuration to be used by this record reader."
  [& {:keys [rr conf]}]
  (doto rr
    (.setConf conf)))

(defn get-listeners
  "Get the record listeners for this record reader."
  [rr]
  (.getListeners rr))

(defn set-listeners
  "Set the record listeners for this record reader."
  [& {:keys [rr listeners]}]
  (doto rr
    (.setListeners listeners)))

(defn next-seq
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

(comment
;; will implement when needed
(ImageRecordReader. )
;;https://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/ImageRecordReader.html
(VideoRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/VideoRecordReader.html
(LibSvmRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/misc/LibSvmRecordReader.html
(SVMLightRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/misc/SVMLightRecordReader.html
(MatlabRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/misc/MatlabRecordReader.html
;; constructor needs args
(ComposableRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/ComposableRecordReader.html
(JacksonRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/jackson/JacksonRecordReader.html
(RegexLineRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/regex/RegexLineRecordReader.html
(RegexSequenceRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/regex/RegexSequenceRecordReader.html
(CSVRegexRecordReader.) ;; constructor needs args
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRegexRecordReader.html
)
