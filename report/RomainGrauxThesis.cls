\NeedsTeXFormat{LaTeX2e}
\newcommand{\classname}{RomainGrauxThesis}
\ProvidesClass{\classname}[2017/08/27 v1.6 LaTeXTemplates.com]
\LoadClass{report}

\RequirePackage{graphicx} % Required to include images
\graphicspath{{./}{./docs/images/}{./docs/plots/img/}} % Specifies where to look for included images

\NewDocumentCommand{\thesistitle} { o m }{%
 \IfValueTF{#1}{\def\shorttitle{#1}}{\def\shorttitle{#2}}%
 \def\@title{#2}%
 \def\ttitle{#2}%
}

\DeclareDocumentCommand{\author}{m}{\newcommand{\authorname}{#1}\renewcommand{\@author}{#1}}
\NewDocumentCommand{\supervisor}{m}{\newcommand{\supname}{#1}}
\NewDocumentCommand{\university}{m}{\newcommand{\univname}{#1}}
\NewDocumentCommand{\lab}{m}{\newcommand{\labname}{#1}}
\NewDocumentCommand{\faculty}{m}{\newcommand{\facname}{#1}}
\NewDocumentCommand{\subject}{m}{\newcommand{\subjectname}{#1}}
\NewDocumentCommand{\keywords}{m}{\newcommand{\keywordnames}{#1}}
