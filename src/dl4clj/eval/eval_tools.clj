(ns ^{:doc "Tools for evaluation and rendering ROC and ROCMultiClass results
see: https://deeplearning4j.org/doc/org/deeplearning4j/evaluation/EvaluationTools.html"}
    dl4clj.eval.eval-tools
  (:import [org.deeplearning4j.evaluation EvaluationTools])
  (:require [clojure.java.io :as io]))

(defn export-roc-charts-to-html-file
  "Given a ROC or ROCMultiClass chart, export the chart to an html file"
  [& {:keys [roc file-path]}]
  (.exportRocChartsToHtmlFile roc (io/as-file file-path)))

(defn roc-chart-to-html
  "Given a ROC or ROCMultiClass instance, render the ROC chart
  and precision vs. recall charts to a stand-alone HTML file (returned as a String)"
  [& {:keys [roc class-labels]
      :as opts}]
  (if (contains? opts :class-labels)
    (.rocChartToHtml roc class-labels)
    (.rocChartToHtml roc)))
