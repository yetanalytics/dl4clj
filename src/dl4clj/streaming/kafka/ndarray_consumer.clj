(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/kafka/NDArrayConsumer.html"}
    dl4clj.streaming.kafka.ndarray-consumer
  (:import [org.deeplearning4j.streaming.kafka NDArrayConsumer]))

#_(defn new-ndarray-consumer
  "NDArray consumer for receiving ndarrays off of kafka"
  []
  (NDArrayConsumer.))
;; more constructor issues

(defn get-arrays-from-stream
  "Receive an ndarray of arrays from the queue"
  [consumer]
  (.getArrays consumer))

(defn get-as-single-array
  "receive a single ndarray from the que
   - need to test if this gets everything as a single array or a single array form the que"
  [consumer]
  (.getINDArray consumer))

(defn start-consumer!
  "starts the consumer and returns it"
  [consumer]
  (.start consumer))
