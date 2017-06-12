(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/conversion/dataset/package-summary.html"}
    dl4clj.streaming.conversion.datasets
  (:import [org.deeplearning4j.streaming.conversion.dataset
            RecordToDataSet
            CSVRecordToDataSet]))

(defn new-csv-record-to-dataset
  "creates a new instance of csv record to dataset.
  Assumes csv format and converts a batch of records into a record matrix"
  []
  (CSVRecordToDataSet.))

(defn convert-csv-to-ds
  "Converts records in to a dataset

  :records (coll), a collection of collection of writeables.
   - see: datavec.api.writeable and datavec.api.split

  :n-labels (int), the number of labels in the ds

  :converter (this), an instance of new-csv-record-to-dataset"
  [& {:keys [records n-labels converter]}]
  (.convert converter records n-labels))
