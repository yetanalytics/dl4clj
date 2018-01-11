;; Copyright (c) 2014 Engagor
;;
;; The use and distribution terms for this software are covered by the
;; BSD License (http://opensource.org/licenses/BSD-2-Clause)
;; which can be found in the file LICENSE at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

(defproject dl4clj "0.1.0-alpha"
  :description "ports of some DL4J features to Clojure"
  :url "https://github.com/yetanalytics/dl4clj"
  :license {:name "BSD C2"
            :url "http://opensource.org/licenses/BSD-2-Clause"}
  :dependencies [[org.clojure/clojure "1.9.0-beta2"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.1"
                  :exclusions
                  [com.google.guava/guava
                   org.apache.commons/commons-compress]]
                 [org.nd4j/nd4j-native-platform "0.9.1"
                  :exclusions [com.google.guava/guava]]
                 [org.datavec/datavec-api "0.9.1"
                  :exclusions
                  [com.google.guava/guava]]
                 [org.datavec/datavec-spark_2.11 "0.9.1_spark_2"
                  :exclusions
                  [org.apache.commons/commons-lang3
                   com.google.guava/guava
                   commons-net
                   org.scala-lang/scala-reflect
                   org.slf4j/slf4j-api
                   org.scala-lang/scala-library
                   org.apache.commons/commons-compress]]
                 [org.deeplearning4j/dl4j-spark_2.11 "0.9.1_spark_2"
                  :exclusions
                  [org.slf4j/slf4j-api
                   commons-net
                   org.scala-lang/scala-library
                   org.scala-lang/scala-reflect
                   org.apache.commons/commons-lang3
                   com.google.guava/guava]]
                 [org.apache.spark/spark-core_2.11 "2.1.0"
                  :exclusions
                  [commons-net
                   org.apache.commons/commons-lang3
                   org.slf4j/slf4j-api]]
                 [org.clojure/core.match "0.3.0-alpha5"]
                 [cheshire "5.7.1"]])
