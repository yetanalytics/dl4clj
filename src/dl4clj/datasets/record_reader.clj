(ns dl4clj.datasets.record-reader
  (:import [org.datavec.api.records.reader
            BaseRecordReader
            RecordReader
            SequenceRecordReader]
           [org.datavec.image.recordreader BaseImageRecordReader
            ImageRecordReader]
           [org.datavec.api.records.reader.impl.collection CollectionRecordReader]
           [org.datavec.api.conf Configuration]
           [org.datavec.api.conf Configurable]
           [org.datavec.api.records.reader.impl FileRecordReader]
           [org.datavec.api.records.reader.factory RecordReaderFactory
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
           [org.datavec.api.records.reader.impl.regex RegexSequenceRecordReader]
           ))

;;erros
(BaseRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/BaseImageRecordReader.html
(BaseImageRecordReader.)

;; working constructors
(ImageRecordReader. )
;;https://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/ImageRecordReader.html
(Configuration.)
(FileRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/FileRecordReader.html
(LineRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/LineRecordReader.html
;;--has 5 subclasses
(ListStringRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/collection/ListStringRecordReader.html
(VideoRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/VideoRecordReader.html
(CSVRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRecordReader.html
(CSVNLinesSequenceRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVNLinesSequenceRecordReader.html
(LibSvmRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/misc/LibSvmRecordReader.html
(SVMLightRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/misc/SVMLightRecordReader.html
(CSVSequenceRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVSequenceRecordReader.html
(MatlabRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/misc/MatlabRecordReader.html

;; constructor needs args
(CollectionRecordReader.) ;; mainly used for testing
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/collection/CollectionRecordReader.html
(CollectionSequenceRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/collection/CollectionSequenceRecordReader.html
(ComposableRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/ComposableRecordReader.html
(JacksonRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/jackson/JacksonRecordReader.html
(CSVRegexRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRegexRecordReader.html
(RegexLineRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/regex/RegexLineRecordReader.html
(RegexSequenceRecordReader.)
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/regex/RegexSequenceRecordReader.html



;; RecordReader is an interface
;; codec doesnt exist on classpath
