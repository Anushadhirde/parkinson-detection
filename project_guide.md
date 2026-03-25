# Parkinson's Disease Detection: Comprehensive Project Guide

## 1. Project Description
The Parkinson's Disease Detection platform is a modern, accessible web application designed to aid the preliminary screening and early diagnosis of Parkinson's Disease (PD). By utilizing non-invasive voice recordings and acoustic data, the system evaluates specific vocal biomarkers (such as micro-tremors, changes in phonation, and articulatory deficits) through a Machine Learning predictive engine. To ensure scalability, accessibility, and high performance, the system is built with a React.js/Next.js frontend and powered by a secure Supabase backend for user authentication, data storage, and record management. It aims to bridge the gap between patients and preliminary clinical diagnosis by offering instant insights into neurological health.

## 2. Objective
- **Primary Goal:** To develop a highly accurate, non-invasive machine learning-based diagnostic tool for evaluating the likelihood of Parkinson's disease using speech acoustics.
- **Secondary Goal:** To create a dynamic and user-friendly web interface (using Next.js) that enables users and healthcare professionals to easily upload, process, and track voice samples over time.
- **Measurable Goals:** Achieve high predictive accuracy (e.g., >90%), ensure real-time analysis with zero downtime, and securely maintain 100% of user screening histories using Supabase PostgreSQL databases.

## 3. Scope
**In-Scope:**
- Web application interface for user registration, authentication, and securely uploading `.wav` or `.mp3` audio recordings.
- Machine Learning functionality to extract acoustic features (Jitter, Shimmer, HNR, MFCCs) and classify the recording as "Healthy" or "Parkinson's".
- A comprehensive dashboard for users to review past results and track progression.
- Integration with Supabase for backend storage, PostgreSQL querying, and Authentication.

**Out-of-Scope:**
- Providing absolute medical diagnosis or replacing professional in-person medical consultations.
- Analyzing other data formats beyond acoustic speech data (e.g., MRI scans or handwriting analysis) within this iteration.

## 4. Benefits
- **Non-Invasive and Accessible:** Patients do not need to undergo complex clinical exams for initial screening; a simple microphone is sufficient.
- **Cost-Effective & Rapid:** Provides automated, instantaneous results at scale, reducing the financial burden and wait times associated with traditional testing.
- **Telemedicine Ready:** Seamlessly integrates into remote healthcare workflows, allowing doctors to monitor patients remotely.
- **Longitudinal Tracking:** The secure data storage enables tracking the progression or degradation of voice quality over time.

## 5. Research Gap
While numerous studies have validated the use of ML for PD detection, the majority of these solutions exist as isolated local scripts or complex clinical software. 
- **Accessibility:** There is a distinct lack of consumer-facing, web-based applications that democratize these ML models.
- **Holistic Ecosystem:** Existing platforms often lack secure historical tracking and modern, intuitive interfaces. 
- **Integration:** This project fills the gap by packaging cutting-edge diagnostic algorithms into a modern tech stack (React/Next.js & Supabase), merging medical research with scalable web engineering.

## 6. Introduction
Parkinson’s disease is highly characterized by motor symptoms such as tremors, rigidity, and bradykinesia, but vocal impairments (dysphonia) are often among the earliest indicators, presenting in nearly 90% of patients. Early detection is paramount for symptom management. Historically, diagnosis has relied heavily on subjective clinical evaluations. The recent evolution of Machine Learning and Signal Processing allows for the objective identification of sub-perceptual vocal degradation. Integrating this capability into an intuitive web architecture enables proactive healthcare screening directly from a modern browser.

## 7. Abstract
This project presents an end-to-end web application for the early detection of Parkinson's Disease via vocal acoustic analysis. Constructed on a robust Next.js and React.js frontend, it ensures a highly responsive and interactive user experience. The backend relies on Supabase for managing PostgreSQL databases, authentication, and secure audio file storage. Central to the application is a predictive Machine Learning module that parses uploaded voice samples, extracts key frequency and amplitude perturbation features, and calculates the probability of PD progression. This holistic ecosystem serves as an advanced telemedicine tool, prioritizing accuracy, speed, user accessibility, and robust data persistence.

## 8. Hardware/Software Specifications
**Hardware Requirements:**
- **Development Machine:** Minimum 8GB RAM, Intel i5 / AMD Ryzen 5 or higher, 256GB SSD.
- **User Side:** Any device (desktop/mobile) with a modern web browser and a working microphone.

**Software Requirements:**
- **Frontend:** React.js, Next.js (App Router), Tailwind CSS (for styling).
- **Backend & Database:** Supabase (PostgreSQL, Supabase Auth, Supabase Storage).
- **Machine Learning (API microservice):** Python 3.9+, Flask/FastAPI, Scikit-Learn, Pandas, Librosa (for audio processing).
- **Languages:** TypeScript/JavaScript, HTML5, CSS3, Python.
- **Development Tools:** VS Code, Git, Node.js, npm/yarn.

## 9. Literature Survey

| S.No | Title | Author | Year | Key Findings | Limitations | Relevance |
|---|---|---|---|---|---|---|
| 1 | Suitability of dysphonia measurements for telemonitoring of PD | Little et al. | 2007 | Identified PPE and HNR as critical acoustic biomarkers for PD. | Small dataset size. | Foundational features used in our model. |
| 2 | Accurate telemonitoring of PD progression by non-invasive speech | Tsanas et al. | 2010 | Developed a mapping between vocal features and UPDRS score using SVM. | Clinical setting bias. | Guides regression and classification ML logic. |
| 3 | Collection and analysis of a Parkinson speech dataset | Sakar et al. | 2013 | Extensive comparison of different phonations and sustained vowels. | Redundant feature sets. | Basis for selecting sustained vowel processing. |
| 4 | Hybrid intelligent system for accurate detection of PD | Hariharan | 2014 | PCA and Neural Networks yield up to 98% accuracy. | High computational cost. | Encourages dimensionality reduction in our ML API. |
| 5 | Discrimination of PD using MFCCs | Benba et al. | 2016 | Mel-frequency cepstral coefficients are highly resilient to noise. | Limited to male/female stratification. | Justifies audio processing pipeline. |
| 6 | Deep Learning-Based PD Classification Using Vocal Feature Sets | Gunduz | 2019 | CNN architectures perform well on raw spectograms. | Requires immense training data. | Context for algorithm selection vs standard SVM/RF. |
| 7 | Detection of PD from speech using deep learning | Zhang et al. | 2017 | End-to-end deep learning bypasses manual feature extraction. | Black-box model nature. | Contrast to our transparent feature-based model. |
| 8 | Healthcare monitoring system for PD prediction using ML | Ali et al. | 2019 | Proves the viability of IoT and web environments for PD testing. | Security/data privacy issues. | Demonstrates the need for proper web integration. |
| 9 | Comparative analysis of predictive models for PD | Haq et al. | 2020 | Random Forest and SVM consistently outperform simple logic trees. | Varies greatly based on data scaling. | Validates using standard Scikit-Learn ensembles. |
| 10 | Automatic evaluation of dysarthria in PD | Vásquez-Correa | 2018 | Analyzed connected speech rather than just sustained vowels. | Hard to parse background noise. | Useful for future scope expansions. |
| 11 | Addressing voice recording protocols for PD diagnosis | Naranjo et al. | 2016 | Standardizing microphone distance and environment limits variance. | Difficult to enforce. | Guides UI instructions for the user. |
| 12 | ML approaches for PD detection from voice frequencies | Rahman et al. | 2021 | Emphasizes recursive feature elimination (RFE). | Prone to overfitting on small sets. | Influences our feature engineering methodology. |
| 13 | Predicting severity of PD using deep learning | Grover et al. | 2018 | Model maps acoustics directly to motor scores effectively. | Clinical validation pending. | Contextualizes disease severity estimation. |
| 14 | Feature selection for PD detection based on speech | Almeida et al. | 2019 | Demonstrates that fewer, robust features often yield better generalized models. | Ignored novel spectral markers. | Highlights the importance of the data ingestion module. |
| 15 | Robust AI framework for telemedicine monitoring of PD | Mei et al. | 2021 | Highlights cloud infrastructure for handling diagnostic loads. | High latency in legacy APIs. | Directly inspires our Next.js + Supabase cloud approach. |

## 10. Methodology
The development follows an agile, modern web-development approach integrated with an ML API.
1. **Data Ingestion via UI:** The user signs into the Next.js app (auth via Supabase), navigates to the upload screen, and records/uploads a voice sample.
2. **Audio Preprocessing & Storage:** The Next.js client sanitizes the file and uploads the `.wav` object directly to a secure Supabase Storage bucket.
3. **API Hook / ML Trigger:** The client immediately calls a Python backend endpoint (API), passing the audio file URL. 
4. **Feature Extraction:** The Python API fetches the file, uses libraries like `librosa` and `parselmouth` to extract acoustic markers (Shimmer, Jitter, MFCCs).
5. **Model Inference:** The extracted feature vector is fed into a pre-trained ML model (e.g., Random Forest or SVM) to obtain a prediction and confidence score.
6. **Database Write:** The prediction result is saved to the Supabase PostgreSQL database associated with the user's ID to maintain a history log.
7. **Result Delivery:** The application retrieves the result and dynamically renders a visually appealing dashboard showing the outcome to the user.

*(For the presentation, you would depict this visually as a flow diagram: User PC -> Next.js Next/React Client -> Supabase Storage -> ML Python Server -> Supabase PostgreSQL DB -> Next.js Client Update).*

## 11. Module Description
- **1. Authentication & Profiling Module:** Powered by Supabase Auth; handles sign-ups, secure logins (JWT tokens), profile management, and session persistency.
- **2. Audio Processing Module (Frontend):** Next.js components dedicated to handling real-time microphone recording and file drag-and-drop mechanics. Validates file types and sizes before network execution.
- **3. Machine Learning Microservice:** An isolated backend service responsible for receiving audio, executing DSP (Digital Signal Processing) feature extraction, and outputting classification probabilities.
- **4. Data Storage Module:** Supabase PostgreSQL schemas design that stores user metadata, historical test results, timestamps, and confidence intervals securely.
- **5. Analytical Dashboard Module:** The UI module responsible for fetching historical data and presenting it effectively to the user via charts, badges, and progress indicators.

## 12. Project Screenshots
*(In your presentation, insert actual screenshots here with the following descriptions)*
1. **Landing Page:** A premium, modern, medically themed hero section explaining the tool's benefits with a "Get Started" call-to-action.
2. **Authentication Screen:** A clean interface featuring email/password and social login options for users to access their dashboard.
3. **Audio Input/Upload Screen:** Shows an interactive drag-and-drop zone and a built-in UI audio recorder with visual sound wave feedback.
4. **Analysis Loading State:** Displays a sleek spinning loader or progress bar while the backend ML model processes the acoustic features.
5. **Results Interface:** A clear, informative card showing "Healthy" or "At Risk," confidence percentages, and a breakdown of analyzed vocal features.
6. **Patient History Dashboard:** A tabular and graphical view of past screenings showing dates, audio files, and trend lines of disease progression.

## 13. Conclusion
The comprehensive Parkinson's Disease Detection system successfully aligns machine learning diagnostics with modern web paradigms. By architecting the application using Next.js and Supabase, the system provides an accessible, rapid, and scalable platform that significantly lowers the barrier for initial neurological screening. The project proves that sophisticated medical models can be effectively deployed via user-friendly telemedicine interfaces, opening avenues for remote healthcare and continuous, non-invasive patient monitoring. Future scope may include expanding to multimodal analysis (combining video tracking for tremors) and deploying native mobile applications.

## 14. References
1. Little, M.A., et al. (2007). *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.* IEEE Transactions on Biomedical Engineering, 56(4), 1015-1022.
2. Sakar, B.E., et al. (2013). *Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings.* IEEE Journal of Biomedical and Health Informatics, 17(4), 828-834.
3. Tsanas, A., et al. (2010). *Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests.* IEEE Transactions on Biomedical Engineering, 57(4), 884-893.
4. React.js and Next.js Official Documentation. Vercel Inc.
5. Supabase Official Documentation - Open Source Firebase Alternative.

---
---

## 15. Project File Structure & Explanation

The application follows the modern Next.js 14+ "App Router" philosophy paired with a standard React best-practice directory structure. 

### **Tree View**
```text
parkinson-detection-web/
 ├── package.json
 ├── next.config.mjs
 ├── tsconfig.json
 ├── tailwind.config.ts
 ├── middleware.ts
 ├── .env.local
 ├── .gitignore
 ├── /public
 │   ├── logo.svg
 │   └── bg-hero.webp
 ├── /app
 │   ├── layout.tsx
 │   ├── page.tsx
 │   ├── globals.css
 │   ├── (auth)
 │   │   ├── login/page.tsx
 │   │   └── register/page.tsx
 │   ├── dashboard/
 │   │   ├── page.tsx
 │   │   └── history/page.tsx
 │   └── detect/
 │       └── page.tsx
 ├── /components
 │   ├── /ui
 │   │   ├── Button.tsx
 │   │   ├── Input.tsx
 │   │   ├── Card.tsx
 │   │   └── Navbar.tsx
 │   ├── AudioUploader.tsx
 │   ├── ResultBadge.tsx
 │   └── HistoryTable.tsx
 ├── /lib
 │   ├── supabaseClient.ts
 │   └── utils.ts
 └── /types
     └── index.d.ts
```

### **Detailed File Explanations**

#### Configuration & Root Files
- **File Name:** `package.json`
  - **Location:** `/package.json`
  - **Purpose:** Manages the project's Node.js metadata, including installed dependencies, project version, and execution scripts (e.g., `npm run dev`).
  - **Key Functions/Components:** `scripts` object, `dependencies` (next, react, @supabase/supabase-js, tailwindcss).
  - **Dependencies:** Node environment. 
  - **Role:** The backbone of the Node ecosystem; ensures any developer can rebuild the environment by running `npm install`.

- **File Name:** `next.config.mjs`
  - **Location:** `/next.config.mjs`
  - **Purpose:** Next.js compiler and bundle settings.
  - **Key Functions/Components:** Configures image domains, experimental features, and API routing behaviors.
  - **Role:** Controls the build output and development server behaviors of the Next.js app.

- **File Name:** `tsconfig.json`
  - **Location:** `/tsconfig.json`
  - **Purpose:** Configuration file for the TypeScript compiler.
  - **Role:** Enforces strict typing rules natively, establishing code reliability and preventing runtime errors before they happen.

- **File Name:** `tailwind.config.ts`
  - **Location:** `/tailwind.config.ts`
  - **Purpose:** Dictates the design system configuration for Tailwind CSS.
  - **Key Functions/Components:** Defines custom medical color palettes (e.g., primary blue, danger red), custom fonts (e.g., Inter), and responsive breakpoints.
  - **Role:** Acts as the single source of truth for the application's entire visual aesthetic.

- **File Name:** `middleware.ts`
  - **Location:** `/middleware.ts`
  - **Purpose:** Next.js Edge middleware primarily used for Route Protection.
  - **Key Functions/Components:** Intercepts requests, checks for a valid Supabase auth session token, and redirects unauthenticated users away from `/dashboard` to `/login`.
  - **Dependencies:** Supabase Auth Helpers.
  - **Role:** The gatekeeper of the application's secure pages.

- **File Name:** `.env.local`
  - **Location:** `/.env.local`
  - **Purpose:** Securely stores environment variables.
  - **Key Functions/Components:** `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, ML_API_URL.
  - **Role:** Protects sensitive connection strings from being exposed to GitHub or the public internet.

#### The App Router (Routing and Pages)
- **File Name:** `layout.tsx`
  - **Location:** `/app/layout.tsx`
  - **Purpose:** The root layout that encases the entire application.
  - **Key Functions/Components:** `RootLayout` functional component. It injects global fonts, the ` Navbar` component, and standard HTML `<head>` tags.
  - **Dependencies:** React, `/app/globals.css`.
  - **Role:** Ensures the UI remains consistent (like headers and footers) across every single page.

- **File Name:** `page.tsx` (Root)
  - **Location:** `/app/page.tsx`
  - **Purpose:** The landing/home page of the website.
  - **Key Functions/Components:** Hero section, application benefits banner, and Call to Action routing to `/detect`.
  - **Role:** Marketing and primary user onboarding.

- **File Name:** `globals.css`
  - **Location:** `/app/globals.css`
  - **Purpose:** Global stylesheet injecting Tailwind base utilities.
  - **Role:** Resets browser defaults and primes the styling engine.

- **File Name:** `login/page.tsx` & `register/page.tsx`
  - **Location:** `/app/(auth)/login/page.tsx` ... 
  - **Purpose:** Renders the authentication interfaces.
  - **Key Functions/Components:** Forms that capture email/password and dispatch them to the `supabase.auth.signInWithPassword()` function.
  - **Role:** Controls user entry into the system. The `(auth)` folder is a Route Group that logically groups them without affecting the URL path.

- **File Name:** `page.tsx` (Dashboard)
  - **Location:** `/app/dashboard/page.tsx`
  - **Purpose:** The primary authenticated home screen.
  - **Key Functions/Components:** Fetches the active user's name from Supabase, summarizes their profile, and provides quick-access buttons to start a new test.
  - **Role:** The control hub for documented users.

- **File Name:** `history/page.tsx`
  - **Location:** `/app/dashboard/history/page.tsx`
  - **Purpose:** Displays tabular history of past screenings.
  - **Key Functions/Components:** `useEffect` hook to run a `SELECT * FROM predictions WHERE user_id = X` query against Supabase PostgreSQL and feed the data into `<HistoryTable />`.
  - **Role:** Provides the long-term monitoring and tracking benefit defined in the project scope.

- **File Name:** `page.tsx` (Detect)
  - **Location:** `/app/detect/page.tsx`
  - **Purpose:** The core functional page where the medical screening takes place.
  - **Key Functions/Components:** Orchestrates the `<AudioUploader />` component, manages the `isLoading` state during the API request, and conditionally renders the `<ResultBadge />` upon success.
  - **Dependencies:** Python ML API endpoint.
  - **Role:** The most critical page of the web app; connects the user frontend interface to the machine learning processing.

#### Components
- **Location:** `/components/ui/` (`Button.tsx`, `Input.tsx`, `Card.tsx`, `Navbar.tsx`)
  - **Purpose:** Reusable, stateless generic UI building blocks.
  - **Key Functions/Components:** Built with dynamic Tailwind classes (e.g., using `clsx` or `tailwind-merge`) to allow size and color variations.
  - **Role:** Enforces DRY (Don't Repeat Yourself) principles and maintains a cohesive visual system.

- **File Name:** `AudioUploader.tsx`
  - **Location:** `/components/AudioUploader.tsx`
  - **Purpose:** UI block for accepting `.wav` files via drag-and-drop or file system dialogue.
  - **Key Functions/Components:** Uses HTML5 Audio API to playback the selected file before upload. Manages a `File` state object.
  - **Role:** Ensures data ingestion is seamless and error-free before hitting the backend.

- **File Name:** `ResultBadge.tsx`
  - **Location:** `/components/ResultBadge.tsx`
  - **Purpose:** Visual representation of the prediction response from the ML model.
  - **Key Functions/Components:** Conditionally renders a red alert UI for "At Risk" and a green safe UI for "Healthy", alongside percentage confidence bars.
  - **Role:** Ensures complex ML outputs are highly comprehensible to laypeople.

- **File Name:** `HistoryTable.tsx`
  - **Location:** `/components/HistoryTable.tsx`
  - **Purpose:** Renders a clean, accessible HTML table mapping array data into rows.
  - **Role:** Separation of concerns; cleans up the dashboard layout by keeping table logic isolated.

#### Libraries and Types
- **File Name:** `supabaseClient.ts`
  - **Location:** `/lib/supabaseClient.ts`
  - **Purpose:** Instantiates the global singleton connection to the database.
  - **Key Functions/Components:** `createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)`
  - **Role:** Used universally by any component or page that needs to read/write data or check auth state. Prevents redundant API connections.

- **File Name:** `utils.ts`
  - **Location:** `/lib/utils.ts`
  - **Purpose:** Exports generic helper functions.
  - **Key Functions/Components:** Functions like `formatDate(rawDateString)` or `cn()` for Tailwind class merging.
  - **Role:** Cleans up component files by abstracting repeatable Javascript logic.

- **File Name:** `index.d.ts`
  - **Location:** `/types/index.d.ts`
  - **Purpose:** Defines global TypeScript interfaces.
  - **Key Functions/Components:** Defines interfaces like `interface UserProfile { id: string, email: string }` and `interface PredictionResult { id: string, date: string, status: string, confidence: number }`.
  - **Role:** Guarantees data structures passed between database, API, and components are safe and predictable.

#### Public Assets
- **Location:** `/public/`
  - **Purpose:** Serves static assets directly to the browser without Webpack/Vite processing.
  - **Files:** `logo.svg` (Project Brand), `bg-hero.webp` (Optimized aesthetic background).
  - **Role:** Improves loading time for high-value visual assets.
