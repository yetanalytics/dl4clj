(ns ^{:doc "see https://deeplearning4j.org/etl-userguide"}
    dl4clj.datavec.api.records.readers
  (:import [org.datavec.api.records.reader
            BaseRecordReader
            RecordReader
            SequenceRecordReader]
           [org.datavec.image.recordreader
            BaseImageRecordReader
            ImageRecordReader]
           [org.datavec.api.records.reader.impl.collection
            CollectionRecordReader]
           [org.datavec.api.conf Configuration]
           [org.datavec.api.conf Configurable]
           [org.datavec.api.records.reader.impl
            FileRecordReader]
           [org.datavec.api.records.reader.factory
            RecordReaderFactory
            RecordWriterFactory]
           [java.io Closeable Serializable]
           [org.datavec.api.records.reader.impl.collection
            CollectionSequenceRecordReader ListStringRecordReader]
           [org.datavec.api.records.reader.impl ComposableRecordReader LineRecordReader]
           [org.datavec.api.records.reader.impl.jackson JacksonRecordReader]
           [java.lang AutoCloseable]
           [org.datavec.image.recordreader VideoRecordReader]
           [org.datavec.api.records.reader.impl.csv CSVRecordReader
            CSVNLinesSequenceRecordReader CSVRegexRecordReader]
           [org.datavec.api.records.reader.impl.misc LibSvmRecordReader]
           [org.datavec.api.records.reader.impl.regex RegexLineRecordReader]
           [org.datavec.api.records.reader.impl.misc SVMLightRecordReader]
           [org.datavec.api.records.reader.impl.csv CSVSequenceRecordReader]
           [org.datavec.api.records.reader.impl.misc MatlabRecordReader]
           [org.datavec.api.records.reader.impl.regex RegexSequenceRecordReader]))

;; build in the ability to make an arbritary record reader using gen-class

(defn record-type
  "dispatch fn for record-reader"
  [opts]
  (first (keys opts)))

(defmulti record-reader
  "Multimethod that builds a record reader based on the supplied type and opts"
  record-type)

(defmethod record-reader :csv-nlines-seq-rr
  [opts]
  (let [config (:csvn-lines-seq-rr opts)
        {skip-lines :skip-num-lines
         delim :delimiter
         n-lines-per-seq :n-lines-per-seq} config]
   (cond (and (true? skip-lines) (true? delim) (true? n-lines-per-seq))
         (CSVNLinesSequenceRecordReader. n-lines-per-seq skip-lines delim)
         (true? n-lines-per-seq)
         (CSVNLinesSequenceRecordReader. n-lines-per-seq)
         :else
         (CSVNLinesSequenceRecordReader. ))))

(defmethod record-reader :csv-rr
  [opts]
  (let [config (:csv-rr opts)
        {skip-lines :skip-num-lines
         delim :delimiter
         strip-quotes :strip-quotes} config]
   (cond (and (true? skip-lines) (true? delim) (true? strip-quotes))
        (CSVRecordReader. skip-lines delim strip-quotes)
        (and (true? skip-lines) (true? delim))
        (CSVRecordReader. skip-lines delim)
        (true? skip-lines)
        (CSVRecordReader. skip-lines)
        :else
        (CSVRecordReader. ))))

(defmethod record-reader :csv-seq-rr
  [opts]
  (let [config (:csv-seq-rr opts)
        {skip-lines :skip-num-lines
         delim :delimiter} config]
    (cond (and (true? skip-lines) (true? delim))
          (CSVSequenceRecordReader. skip-lines delim)
          (true? skip-lines)
          (CSVSequenceRecordReader. skip-lines)
          :else
          (CSVSequenceRecordReader. ))))


(defmethod record-reader :file-rr
  [opts]
  (FileRecordReader.))

(defmethod record-reader :line-rr
  [opts]
  (LineRecordReader.))

(defmethod record-reader :list-string-rr
  [opts]
  ;; for initialization, only accepts a ListStringInputSplit
  (ListStringRecordReader.))


(defn close [rr]
  (doto rr
    (.close)))

(defn get-conf [rr]
  (.getConf rr))

(defn get-labels [rr]
  (.getLabels rr))

(defn has-next? [rr]
  (.hasNext rr))

(defn initialize
  ([rr input-split]
   (.initialize rr input-split))
  ([rr input-split conf]
   (.initialize rr conf input-split)))

(defn load-from-meta-data
  [rr record-meta-data]
  (.loadFromMetaData rr record-meta-data))

(defn next-data-record
  [rr]
  (.next rr))

(defn next-record
  [rr]
  (.nextRecord rr))

(defn record
  [rr uri data-in-stream]
  (.record rr uri data-in-stream ))

(defn reset
  [rr]
  (doto rr
    (.reset)))

(defn set-conf
  [rr conf]
  (doto rr
    (.setConf conf)))

(defn get-listeners
  [rr]
  (.getListeners rr))

(defn set-listeners
  [rr listeners]
  (doto rr
    (.setListeners listeners)))

(defn load-seq-from-meta-data
  [rr rr-meta-data]
  (.loadSequenceFromMetaData rr rr-meta-data))

(defn next-seq
  [rr]
  (.nextSequence rr))

(defn sequence-record
  [rr & {:keys [uri data-in-stream]}]
  (if (and (true? uri) (true? data-in-stream))
    (.sequenceRecord rr uri data-in-stream)
    (.sequenceRecord rr)))


(record-reader :csv-rr)

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
