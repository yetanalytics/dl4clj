(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/kafka/NDArrayConsumer.html"}
    dl4clj.streaming.kafka.ndarray-consumer
  (:import [org.deeplearning4j.streaming.kafka NDArrayConsumer]
           [org.apache.camel.impl DefaultCamelContext]
           [org.apache.camel.impl DefaultConsumerTemplate]))

(defn new-ndarray-consumer
  "NDArray consumer for receiving ndarrays off of kafka

  :topic-name (str), the name of the kafka topic

  :kafka-uri (str), the uri to a kafka stream containing NDArrays"
  [& {:keys [topic-name kafka-uri]}]
  (let [camel-context (DefaultCamelContext.)
        consumer-template (DefaultConsumerTemplate. camel-context)]
   (NDArrayConsumer. camel-context consumer-template topic-name
                     kafka-uri false)))

(defn start-consumer!
  "starts the consumer and returns it"
  [consumer]
  (.start consumer))

(defn get-arrays-from-stream
  "Receive an ndarray of arrays from the queue"
  [consumer]
  (.getArrays consumer))

(defn get-as-single-array
  "receive a single ndarray from the que
   - need to test if this gets everything as a single array or a single array form the que"
  [consumer]
  (.getINDArray consumer))
