import { useState } from "react";
import Header from "./components/Header.jsx";
import SessionGate from "./components/SessionGate.jsx";
import Dashboard from "./components/Dashboard.jsx";

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [enterTime, setEnterTime] = useState(null);

  function handleEnter(sid, time) {
    setSessionId(sid);
    setEnterTime(time);
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-[1080px] flex-col gap-[20px] px-4 pb-[72px] pt-5 min-[720px]:gap-[28px] min-[720px]:px-8 min-[720px]:pb-[96px] min-[720px]:pt-8">
      <Header />
      {!sessionId ? (
        <SessionGate onEnter={handleEnter} />
      ) : (
        <Dashboard sessionId={sessionId} enterTime={enterTime} />
      )}
    </div>
  );
}