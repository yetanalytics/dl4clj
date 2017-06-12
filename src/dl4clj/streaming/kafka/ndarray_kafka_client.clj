(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/kafka/NDArrayKafkaClient.html"}
    dl4clj.streaming.kafka.ndarray-kafka-client
  (:import [org.deeplearning4j.streaming.kafka NDArrayKafkaClient]))

#_(defn new-ndarray-kafka-client
  ""
  []
    (NDArrayKafkaClient.))
;; again non working constructor
;; I think it could be bc of the annontations, even tho the class should pull them in...

(defn create-kafka-consumer
  [client]
  (.createConsumer client))

(defn create-kafka-publisher
  [client]
  (.createPublisher client))
