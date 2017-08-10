(ns ^{:doc "see https://deeplearning4j.org/etl-userguide"}
    dl4clj.datasets.record-readers
  (:import [org.datavec.api.records.reader BaseRecordReader RecordReader SequenceRecordReader]
           [org.datavec.image.recordreader BaseImageRecordReader ImageRecordReader]
           [org.datavec.api.records.reader.impl.collection CollectionRecordReader]
           [org.datavec.api.records.reader.impl FileRecordReader]
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
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; replace contains-many? with core.match
(defmulti record-reader
  "Multimethod that builds a record reader based on the supplied type and opts"
  generic-dispatching-fn)

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

  NOTE: This record reader is for saying, there are multiple time series which should
        be considered together and classified by a single label (for classification)

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
