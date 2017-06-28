(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/kafka/NDArrayKafkaClient.html"}
    dl4clj.streaming.kafka.ndarray-kafka-client
  (:import [org.deeplearning4j.streaming.kafka NDArrayKafkaClient
            NDArrayKafkaClient$NDArrayKafkaClientBuilder]
           [org.apache.camel.impl DefaultCamelContext]))


;; still not sure how to invoke constructor

#_(defn new-ndarray-kafka-client
  ""
  [& {:keys [kafka-uri zoo-keeper-connection kafka-topic]}]
  (let [camel-context (DefaultCamelContext.)]
    (NDArrayKafkaClient. kafka-uri zoo-keeper-connection camel-context kafka-topic "ndarraytype")))
