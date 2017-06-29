(ns dl4clj.eval.meta
  (:import [org.deeplearning4j.eval.meta Prediction]))
;; placehold, constructor is acting weird
;; look into src before core branch
(defn get-record-meta-data
  "Convenience method for getting the record meta data as a particular class"
  [& {:keys [prediction desired-class]}]
  (.getRecordMetaData prediction desired-class))
