;; Copyright (c) 2014 Engagor
;;
;; The use and distribution terms for this software are covered by the
;; BSD License (http://opensource.org/licenses/BSD-2-Clause)
;; which can be found in the file LICENSE at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

(defproject dl4clj "0.0.2-SNAPSHOT"
  :description "ports of some DL4J examples to Clojure"
  :url "https://github.com/engagor/dl4clj"
  :license {:name "BSD C2"
            :url "http://opensource.org/licenses/BSD-2-Clause"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha16"]
                 [org.deeplearning4j/deeplearning4j-core "0.8.0"]
                 [org.nd4j/nd4j-native-platform "0.8.0"]
                 [org.datavec/datavec-api "0.8.0"]
                 [org.datavec/datavec-spark_2.11 "0.8.0_spark_2"]
                 [org.deeplearning4j/dl4j-spark_2.11 "0.8.0_spark_2"]
                 [org.apache.spark/spark-core_2.11 "2.1.0"]
                 #_[org.deeplearning4j/dl4j-streaming_2.11 "0.8.0_spark_2"]
                 [org.clojure/core.match "0.3.0-alpha5"]
                 [cheshire "5.7.1"]])
