(ns dl4clj.streaming.kafka.uri-builder
  (:import [org.deeplearning4j.streaming.kafka KafkaUriBuilder]))

(defn create-kafka-uri
  [& {:keys [kafka-broker consuming-topic group-id]}]
  (str "kafka://" kafka-broker "?" "topic=" consuming-topic "&groupId=" group-id))

#_(= (java.lang.String/format "kafka://%s?topic=%s&groupId=%s" (into-array java.lang.String ["broker" "topic" "group-id"]))
   (create-kafka-uri :kafka-broker "broker"
                     :consuming-topic "topic"
                     :group-id "group-id"))
