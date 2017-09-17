(ns ^{:doc "Tools for evaluation and rendering ROC and ROCMultiClass results
see: https://deeplearning4j.org/doc/org/deeplearning4j/evaluation/EvaluationTools.html"}
    dl4clj.eval.api.eval-tools
  (:import [org.deeplearning4j.evaluation EvaluationTools])
  (:require [clojure.java.io :as io]
            [dl4clj.utils :refer [obj-or-code?]]
            [clojure.core.match :refer [match]]))

(defn export-roc-charts-to-html-file
  "Given a ROC or ROCMultiClass chart, export the chart to an html file"
  [& {:keys [roc file-path as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:roc (_ :guard seq?)
           :file-path (:or (_ :guard string?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(EvaluationTools/exportRocChartsToHtmlFile ~roc (io/as-file ~file-path)))
         :else
         (EvaluationTools/exportRocChartsToHtmlFile roc (io/as-file file-path))))

(defn roc-chart-to-html
  "Given a ROC or ROCMultiClass instance, render the ROC chart
  and precision vs. recall charts to a stand-alone HTML file (returned as a String)"
  [& {:keys [roc class-labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:roc (_ :guard seq?)
           :class-labels (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(EvaluationTools/rocChartToHtml ~roc ~class-labels))
         [{:roc _
           :class-labels _}]
         (EvaluationTools/rocChartToHtml roc class-labels)
         [{:roc (_ :guard seq?)}]
         (obj-or-code? as-code? `(EvaluationTools/rocChartToHtml ~roc))
         [{:roc _}]
         (EvaluationTools/rocChartToHtml roc)))
