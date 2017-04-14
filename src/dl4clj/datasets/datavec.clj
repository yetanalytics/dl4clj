(ns dl4clj.datasets.datavec
  (:import [org.deeplearning4j.datasets DataSets]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator
            RecordReaderMultiDataSetIterator$Builder RecordReaderMultiDataSetIterator
            SequenceRecordReaderDataSetIterator]))


(def iris-ds (DataSets/iris))

(def mnist-ds (DataSets/mnist))
