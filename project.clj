(defproject dl4clj "0.1.0-SNAPSHOT"
  :description "ports of some DL4J examples to Clojure"
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.4-rc3.7"]
                 [org.deeplearning4j/deeplearning4j-ui "0.4-rc3.8"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.4-rc3.8"]
                 [org.apache.commons/commons-io "1.3.2"]
                 [org.nd4j/nd4j-jblas "0.4-rc3.6"]            ;; MacBook requirement
                 [org.nd4j/nd4j-x86 "0.4-rc3.5"]              ;; CPU
                 [org.nd4j/canova-api "0.0.0.14"]
                 [org.clojure/data.json "0.2.6"]
                 ;; [com.google.collections/google-collections "1.0"]
                 ]) 
