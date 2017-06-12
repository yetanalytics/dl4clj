(ns dl4clj.streaming.conversion.ndarray
  (:import [org.deeplearning4j.streaming.conversion.ndarray
            RecordToNDArray
            NDArrayRecordToNDArray
            CSVRecordToINDArray]))

(defn new-csv-record-to-indarray
  "Assumes csv format and converts a batch of records into a matrix."
  []
  (CSVRecordToINDArray.))

(defn new-ndarray-record-to-ndarray
  "Assumes all records in the given batch are of type NDArrayWritable.
  It extracts the underlying arrays and returns a concatenated array."
  []
  (NDArrayRecordToNDArray.))

(defn convert-to-ndarray
  "Converts a list of records in to 1d ndarray

  :records (coll), a collection of collection of writables
   - see: datavec.api.writeable and datavec.common.data.ndarray-writable"
  [& {:keys [records converter]}]
  (.convert converter records))
