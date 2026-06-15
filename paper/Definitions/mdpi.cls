%% mdpi.cls -- Minimal MDPI-compatible document class
%% Provides all MDPI-specific commands for single-column journal articles.
%% Based on article class with MDPI formatting conventions.
%% Version: 1.0 (minimal, for ActiveCircuitDiscovery paper)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Definitions/mdpi}[2026/01/01 Minimal MDPI class]

%% -------------------------------------------------------
%% Class options
%% -------------------------------------------------------
\newif\if@preprints\@preprintsfalse
\newif\if@submit\@submittrue
\newif\if@pdftex\@pdftextrue
\newif\if@moreauthors\@moreauthorsfalse
\newif\if@oneauthor\@oneauthorfalse

\DeclareOption{preprints}{\@preprintstrue}
\DeclareOption{submit}{\@submittrue}
\DeclareOption{accept}{\@submitfalse}
\DeclareOption{pdftex}{\@pdftextrue}
\DeclareOption{moreauthors}{\@moreauthorstrue}
\DeclareOption{oneauthor}{\@oneauthortrue}
%% Accept any journal name as option (ignore)
\DeclareOption*{}
\ProcessOptions\relax

%% Load base class: single-column article
\LoadClass[12pt,a4paper]{article}

%% -------------------------------------------------------
%% Required packages
%% -------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{calc}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{float}
\RequirePackage{microtype}
\RequirePackage{xcolor}
\RequirePackage{colortbl}
\RequirePackage{soul}
\RequirePackage{lineno}
\RequirePackage{setspace}
\RequirePackage[left=2cm,right=2cm,top=2.5cm,bottom=2.5cm]{geometry}
\RequirePackage{enumitem}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage[hyphens]{url}
\RequirePackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\RequirePackage{cleveref}
\RequirePackage{upgreek}
\RequirePackage{changepage}
\RequirePackage{etoolbox}
\RequirePackage{mathpazo}
\RequirePackage{lastpage}
\RequirePackage{ifthen}
\RequirePackage{listings}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{mathtools}
\RequirePackage{bm}
\RequirePackage{makecell}
% cite package removed - natbib handles citations

%% -------------------------------------------------------
%% MDPI-specific lengths
%% -------------------------------------------------------
\newlength{\extralength}
\setlength{\extralength}{0cm}
\newlength{\fulllength}
\setlength{\fulllength}{\textwidth}

%% -------------------------------------------------------
%% Internal state storage
%% -------------------------------------------------------
\newcommand{\@mdpititle}{}
\newcommand{\@mdpiauthor}{}
\newcommand{\@mdpiauthornames}{}
\newcommand{\@mdpiaddress}{}
\newcommand{\@mdpicorres}{}
\newcommand{\@mdpiabstract}{}
\newcommand{\@mdpikeyword}{}
\newcommand{\@pubyear}{2026}
\newcommand{\@copyrightyear}{2026}
\newcommand{\@pubvolume}{1}
\newcommand{\@issuenum}{1}
\newcommand{\@articlenumber}{0}
\newcommand{\@datereceived}{}
\newcommand{\@daterevised}{}
\newcommand{\@dateaccepted}{}
\newcommand{\@datepublished}{}

%% -------------------------------------------------------
%% MDPI metadata commands (preamble)
%% -------------------------------------------------------
\newcommand{\@firstpage}{1}
\newcommand{\firstpage}[1]{\renewcommand{\@firstpage}{#1}\setcounter{page}{#1}}
\newcommand{\pubvolume}[1]{\renewcommand{\@pubvolume}{#1}}
\newcommand{\issuenum}[1]{\renewcommand{\@issuenum}{#1}}
\newcommand{\articlenumber}[1]{\renewcommand{\@articlenumber}{#1}}
\newcommand{\pubyear}[1]{\renewcommand{\@pubyear}{#1}}
\newcommand{\copyrightyear}[1]{\renewcommand{\@copyrightyear}{#1}}
\newcommand{\datereceived}[1]{\renewcommand{\@datereceived}{#1}}
\newcommand{\daterevised}[1]{\renewcommand{\@daterevised}{#1}}
\newcommand{\dateaccepted}[1]{\renewcommand{\@dateaccepted}{#1}}
\newcommand{\datepublished}[1]{\renewcommand{\@datepublished}{#1}}
\newcommand{\externaleditor}[1]{}
%\newcommand{\pdfoutput}[1]{} % pdfoutput is already defined by pdftex
\newcommand{\CorrStatement}[1]{}
\newcommand{\longauthorlist}[1]{}
\newcommand{\IsAssociation}[1]{}

%% ORCID icon commands (display only; orcidauthorX are defined by user)
\newcommand{\orcidA}{\textsuperscript{\tiny\textcircled{iD}}}
\newcommand{\orcidB}{\textsuperscript{\tiny\textcircled{iD}}}
\newcommand{\orcidC}{\textsuperscript{\tiny\textcircled{iD}}}

%% Author/affiliation commands
\newcommand{\Title}[1]{\renewcommand{\@mdpititle}{#1}\title{\textbf{#1}}}
\newcommand{\Author}[1]{\renewcommand{\@mdpiauthor}{#1}\author{#1}}
\newcommand{\AuthorNames}[1]{\renewcommand{\@mdpiauthornames}{#1}}
% Override \address if defined by base class
\providecommand{\address}[1]{}
\renewcommand{\address}[1]{\renewcommand{\@mdpiaddress}{#1}}
\newcommand{\corres}[1]{\renewcommand{\@mdpicorres}{#1}}

\newcommand{\firstnote}[1]{}
\newcommand{\secondnote}[1]{}
\newcommand{\thirdnote}[1]{}
\newcommand{\fourthnote}[1]{}
\newcommand{\fifthnote}[1]{}
\newcommand{\sixthnote}[1]{}
\newcommand{\seventhnote}[1]{}
\newcommand{\eighthnote}[1]{}
\newcommand{\simplesumm}[1]{}
\newcommand{\conference}[1]{}

%% Subject classification
\newcommand{\PACS}[1]{}
\newcommand{\MSC}[1]{}
\newcommand{\JEL}[1]{}
\newcommand{\LSID}[1]{}

%% Abstract and keywords (stored for use in \maketitle)
%% Use \mdpiabstract to avoid conflict with abstract environment
\newcommand{\mdpiabstract}[1]{\renewcommand{\@mdpiabstract}{#1}}
% Redefine \abstract as command (not environment) for MDPI preamble use
\let\abstractenv\abstract
\let\endabstractenv\endabstract
\renewcommand{\abstract}[1]{\renewcommand{\@mdpiabstract}{#1}}
\newcommand{\keyword}[1]{\renewcommand{\@mdpikeyword}{#1}}

%% Dataset
\newcommand{\dataset}[1]{}
\newcommand{\datasetlicense}[1]{}
\newcommand{\keycontribution}[1]{}
\newcommand{\encyclopediadef}[1]{}

%% Highlights
\newcommand{\addhighlights}[1]{}

%% -------------------------------------------------------
%% maketitle override
%% -------------------------------------------------------
\renewcommand{\maketitle}{%
  \thispagestyle{empty}%
  \begin{center}
    {\LARGE\bfseries\@mdpititle\par}
    \vspace{1em}
    {\large\@mdpiauthor\par}
    \vspace{0.5em}
    {\small\itshape\@mdpiaddress\par}
    \vspace{0.3em}
    {\small\textbf{Correspondence:}~\@mdpicorres\par}
    \vspace{0.3em}
    \ifthenelse{\equal{\@datereceived}{}}{}{%
      {\footnotesize Received: \@datereceived\quad
      \ifthenelse{\equal{\@daterevised}{}}{}{Revised: \@daterevised\quad}%
      \ifthenelse{\equal{\@dateaccepted}{}}{}{Accepted: \@dateaccepted\quad}%
      \ifthenelse{\equal{\@datepublished}{}}{}{Published: \@datepublished}%
      \par}%
    }
  \end{center}
  \vspace{1em}
  \begin{minipage}{\textwidth}
    \small
    \textbf{Abstract:} \@mdpiabstract
    \vspace{0.5em}

    \noindent\textbf{Keywords:} \@mdpikeyword
  \end{minipage}
  \vspace{1em}
  \setlength{\parindent}{6mm}
}

%% -------------------------------------------------------
%% Page style
%% -------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\itshape Symmetry \textbf{\@pubvolume}, \@articlenumber}
\fancyhead[R]{\footnotesize\thepage\ of \pageref{LastPage}}
\fancyfoot[C]{}
\renewcommand{\headrulewidth}{0pt}

%% -------------------------------------------------------
%% Line numbering (submit mode)
%% -------------------------------------------------------
\if@submit
  \linenumbers
  % lineno provides linenomath environment - no need to redefine
\fi

%% -------------------------------------------------------
%% Section formatting (MDPI-style)
%% -------------------------------------------------------
\setcounter{secnumdepth}{3}

%% -------------------------------------------------------
%% Theorem environments
%% -------------------------------------------------------
\newtheoremstyle{mdpiplain}
  {\topsep}{\topsep}
  {\itshape}{}
  {\bfseries}{.}{ }{}
\theoremstyle{mdpiplain}
\newtheorem{Theorem}{Theorem}
\newtheorem{Lemma}[Theorem]{Lemma}
\newtheorem{Corollary}[Theorem]{Corollary}
\newtheorem{Proposition}[Theorem]{Proposition}
\newtheorem{Characterization}[Theorem]{Characterization}
\newtheorem{Property}[Theorem]{Property}
\newtheoremstyle{mdpidefinition}
  {\topsep}{\topsep}
  {}{}{\bfseries}{.}{ }{}
\theoremstyle{mdpidefinition}
\newtheorem{Problem}[Theorem]{Problem}
\newtheorem{Example}[Theorem]{Example}
\newtheorem{ExamplesandDefinitions}[Theorem]{Examples and Definitions}
\newtheorem{Hypothesis}[Theorem]{Hypothesis}
\newtheorem{Remark}[Theorem]{Remark}
\newtheorem{Definition}[Theorem]{Definition}
\newtheorem{Notation}[Theorem]{Notation}
\newtheorem{Assumption}[Theorem]{Assumption}

%% -------------------------------------------------------
%% Caption formatting
%% -------------------------------------------------------
\captionsetup{font=small,labelfont=bf}

%% -------------------------------------------------------
%% tabularx C column type
%% -------------------------------------------------------
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

%% -------------------------------------------------------
%% Figure/table positioning
%% -------------------------------------------------------
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.10}
\renewcommand{\floatpagefraction}{0.85}

%% -------------------------------------------------------
%% isPreprints command (two-arg conditional)
%% -------------------------------------------------------
\newcommand{\isPreprints}[2]{%
  \if@preprints#1\else#2\fi
}

%% isAPAandChicago, isChicagoStyle, isAPAStyle (bibliography conditionals)
\newcommand{\isAPAandChicago}[2]{#1}
\newcommand{\isChicagoStyle}[2]{#2}
\newcommand{\isAPAStyle}[2]{#2}

%% -------------------------------------------------------
%% End-of-paper MDPI sections (boilerplate commands)
%% -------------------------------------------------------
\newcommand{\authorcontributions}[1]{%
  \vspace{6pt}
  \noindent\textbf{Author Contributions:}~#1
  \vspace{6pt}
}
\newcommand{\funding}[1]{%
  \noindent\textbf{Funding:}~#1
  \vspace{6pt}
}
\newcommand{\institutionalreview}[1]{%
  \noindent\textbf{Institutional Review Board Statement:}~#1
  \vspace{6pt}
}
\newcommand{\informedconsent}[1]{%
  \noindent\textbf{Informed Consent Statement:}~#1
  \vspace{6pt}
}
\newcommand{\dataavailability}[1]{%
  \noindent\textbf{Data Availability Statement:}~#1
  \vspace{6pt}
}
\newcommand{\durcstatement}[1]{%
  \noindent\textbf{Dual-Use Research of Concern (DURC) Statement:}~#1
  \vspace{6pt}
}
\newcommand{\publicinvolvement}[1]{%
  \noindent\textbf{Public Involvement Statement:}~#1
  \vspace{6pt}
}
\newcommand{\guidelinesstandards}[1]{%
  \noindent\textbf{Guidelines and Standards Statement:}~#1
  \vspace{6pt}
}
\newcommand{\useofartificialintelligence}[1]{%
  \noindent\textbf{Use of Artificial Intelligence Statement:}~#1
  \vspace{6pt}
}
\newcommand{\acknowledgments}[1]{%
  \noindent\textbf{Acknowledgments:}~#1
  \vspace{6pt}
}
\newcommand{\conflictsofinterest}[1]{%
  \noindent\textbf{Conflicts of Interest:}~#1
  \vspace{6pt}
}
\newcommand{\entrylink}[1]{}

%% Abbreviations
\newcommand{\abbreviations}[2]{%
  \vspace{6pt}
  \noindent\textbf{#1:}

  #2
  \vspace{6pt}
}

%% References title
\newcommand{\reftitle}[1]{}

%% Publishers note
\newcommand{\PublishersNote}[1]{}

%% -------------------------------------------------------
%% Appendix support
%% -------------------------------------------------------
\newcommand{\appendixtitles}[1]{}
\newcommand{\appendixstart}{}

%% Make appendix sections labeled A, B, C...
\let\oldappendix\appendix
\renewcommand{\appendix}{%
  \oldappendix%
  \renewcommand{\thesection}{\Alph{section}}%
  \renewcommand{\thesubsection}{\Alph{section}\arabic{subsection}}%
  \setcounter{section}{0}%
  \setcounter{equation}{0}%
  \renewcommand{\theequation}{\Alph{section}\arabic{equation}}%
  \setcounter{figure}{0}%
  \renewcommand{\thefigure}{\Alph{section}\arabic{figure}}%
  \setcounter{table}{0}%
  \renewcommand{\thetable}{\Alph{section}\arabic{table}}%
}

%% -------------------------------------------------------
%% Review reports (for Sci journal)
%% -------------------------------------------------------
\newcommand{\reviewreports}[1]{}

%% -------------------------------------------------------
%% Supplement
%% -------------------------------------------------------
\newcommand{\supplementary}[1]{%
  \noindent\textbf{Supplementary Materials:}~#1
}
\newcommand{\linksupplementary}[1]{\url{#1}}

%% -------------------------------------------------------
%% Date macros
%% -------------------------------------------------------
\newcommand{\doinum}[1]{}

%% -------------------------------------------------------
%% Bio/photo of author
%% -------------------------------------------------------
\newcommand{\bio}[2]{%
  \noindent #1 \smallskip

  \noindent #2 \bigskip
}

%% -------------------------------------------------------
%% Hyperref setup override
%% -------------------------------------------------------
\AtBeginDocument{%
  \hypersetup{
    pdftitle={\@mdpititle},
    pdfauthor={\@mdpiauthornames},
  }%
}

\endinput
