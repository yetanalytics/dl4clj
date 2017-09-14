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
  (:require [dl4clj.utils :refer [generic-dispatching-fn obj-or-code?]]
            [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti record-reader
  "Multimethod that builds a record reader based on the supplied type and opts"
  generic-dispatching-fn)

(defmethod record-reader :csv-nlines-seq-rr [opts]
  (let [config (:csv-nlines-seq-rr opts)
        {skip-lines :skip-n-lines
         delim :delimiter
         n-lines-per-seq :n-lines-per-seq} config]
    (match [config]
           [{:skip-n-lines _ :delimiter _ :n-lines-per-seq _}]
           `(CSVNLinesSequenceRecordReader. ~n-lines-per-seq ~skip-lines ~delim)
           [{:n-lines-per-seq _}]
           `(CSVNLinesSequenceRecordReader. ~n-lines-per-seq)
           :else
           `(CSVNLinesSequenceRecordReader.))))

(defmethod record-reader :csv-rr [opts]
  (let [config (:csv-rr opts)
        {skip-lines :skip-n-lines
         delim :delimiter
         strip-quotes :strip-quotes} config]
    (match [config]
           [{:skip-n-lines _ :delimiter _ :strip-quotes _}]
           `(CSVRecordReader. ~skip-lines ~delim ~strip-quotes)
           [{:skip-n-lines _ :delimiter _}]
           `(CSVRecordReader. ~skip-lines ~delim)
           [{:skip-n-lines _}]
           `(CSVRecordReader. ~skip-lines)
           :else
           `(CSVRecordReader.))))

(defmethod record-reader :csv-seq-rr [opts]
  (let [config (:csv-seq-rr opts)
        {skip-lines :skip-n-lines
         delim :delimiter} config]
    (match [config]
           [{:skip-n-lines _ :delimiter _}]
           `(CSVSequenceRecordReader. ~skip-lines ~delim)
           [{:skip-n-lines _}]
           `(CSVSequenceRecordReader. ~skip-lines)
           :else
           `(CSVSequenceRecordReader.))))

(defmethod record-reader :file-rr [opts]
  `(FileRecordReader.))

(defmethod record-reader :line-rr [opts]
  `(LineRecordReader.))

(defmethod record-reader :list-string-rr [opts]
  `(ListStringRecordReader.))

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
  :skip-n-lines (int) number of lines to skip
  :delimiter (str) the delimiter seperating values
  :n-lines-per-seq (int) the number of lines which compose a single series
  :as-code? (boolean), return java object or code for creating it

  NOTE: This record reader is for saying, there are multiple time series which should
        be considered together and classified by a single label (for classification)

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVNLinesSequenceRecordReader.html"
  [& {:keys [skip-n-lines delimiter n-lines-per-seq as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (record-reader {:csv-nlines-seq-rr opts})]
    (obj-or-code? as-code? code)))

(defn new-csv-record-reader
  "Simple csv record reader

  args are:
  :skip-n-lines (int) number of lines to skip
  :delimiter (str) the delimiter seperating values
  :strip-quotes (str) the quote to strip
  :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRecordReader.html"
  [& {:keys [skip-n-lines delimiter strip-quotes as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (record-reader {:csv-rr opts})]
    (obj-or-code? as-code? code)))

(defn new-csv-seq-record-reader
  "This reader is intended to read sequences of data in CSV format,
   where each sequence is defined in its own file (and there are multiple files)
   Each line in the file represents one time step

   args are:
   :skip-n-lines (int) number of lines to skip
   :delimiter (str), the delimiter seperating values
   :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVSequenceRecordReader.html"
  [& {:keys [skip-n-lines delimiter as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (record-reader {:csv-seq-rr opts})]
    (obj-or-code? as-code? code)))

(defn new-file-record-reader
  "File reader/writer, no args required

  :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/FileRecordReader.html"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (record-reader {:file-rr {}})]
    (obj-or-code? as-code? code)))

(defn new-line-record-reader
  "Reads files line by line, no args required

  :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/LineRecordReader.html"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (record-reader {:line-rr {}})]
    (obj-or-code? as-code? code)))

(defn new-list-string-record-reader
  "Iterates through a list of strings return a record.
  Only accepts a list-string-input-split as input during initialization

  no args needed to call the constructor

  :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/collection/ListStringRecordReader.html"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (record-reader {:list-string-rr {}})]
    (obj-or-code? as-code? code)))
